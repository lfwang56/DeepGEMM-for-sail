"""Reproduce the selected EP32/EP64 BF16/FP8 routed MoE layer recipes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ep', type=int, choices=(32, 64), default=32, help='Devices in one physical supernode; default32')
    parser.add_argument('--model', choices=('mok', 'qwen'), required=True)
    parser.add_argument('--precision', choices=('bf16', 'fp8'), required=True)
    parser.add_argument('--dry-run', action='store_true', help='Print the exact environment and command without importing Torch')
    parser.add_argument('--validate-configuration', action='store_true', help='Run benchmark argument validation without initializing a device')
    parser.add_argument('--profile', action='store_true')
    for name in ('warmup', 'iterations', 'repeats'):
        parser.add_argument('--' + name, type=int)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / 'docs/ppu_moe_training_recipes.json').read_text())
    recipe = next(r for r in config['recipes'] if r['ep'] == args.ep and r['model'] == args.model and r['precision'] == args.precision)
    lut = root / recipe['lut_directory']
    for filename, expected in recipe['lut_sha256'].items():
        if hashlib.sha256((lut / filename).read_bytes()).hexdigest() != expected:
            raise ValueError('Selected DW LUT changed: ' + filename)
    environment = {**config['common_environment'], **recipe['environment'], 'SAIL_DEEPGEMM_LUT_PATH': str(lut)}
    arguments = list(recipe['benchmark_arguments'])
    for name in ('warmup', 'iterations', 'repeats'):
        value = getattr(args, name)
        if value is not None:
            if value < (0 if name == 'warmup' else 1):
                parser.error(name + ' is outside the benchmark range')
            arguments[arguments.index('--' + name) + 1] = str(value)
    if args.profile:
        arguments.append('--profile')
    if args.validate_configuration:
        arguments.append('--validate-configuration')
    benchmark = root / 'tests/benchmark_native_moe.py'
    resolved = dict(case='selected_moe_training_recipe', recipe=recipe['id'], expected_world_size=recipe['ep'],
                    dw='BF16 operands with FP32 accumulation/output', communication='BF16 direct peer payloads',
                    measurement=recipe['measurement'],
                    environment=environment, command=[sys.executable, str(benchmark), *arguments])
    if args.dry_run:
        print(json.dumps(resolved, indent=2))
        return
    if not args.validate_configuration and int(os.environ.get('WORLD_SIZE', '1')) != recipe['ep']:
        parser.error(f"This recipe requires WORLD_SIZE={recipe['ep']}. Launch it with torchrun on {recipe['ep'] // config['target']['devices_per_host']} eight-device hosts in one physical supernode.")
    os.environ.update(environment)
    if int(os.environ.get('RANK', '0')) == 0:
        print(json.dumps(resolved), flush=True)
    sys.argv = [str(benchmark), *arguments]
    runpy.run_path(str(benchmark), run_name='__main__')


if __name__ == '__main__':
    main()
