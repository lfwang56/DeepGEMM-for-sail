"""Check Gitless source versioning and reject a different selected PPU SDK."""
import json
from pathlib import Path
import runpy
import tempfile

namespace = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'setup.py'))
version = namespace['get_package_version']
with tempfile.TemporaryDirectory() as temporary:
    version.__globals__['current_dir'] = temporary
    metadata = Path(temporary) / 'PKG-INFO'
    expected = '1.1.0.dev50+v1.1.0.ppu2.2.0.g1234abcd'
    def check(name, value, sdk, accepted):
        metadata.write_text(f'Metadata-Version: 2.1\nName: {name}\nVersion: {value}\n')
        try:
            result = version(sdk)
        except RuntimeError:
            assert not accepted, (name, value, sdk)
        else:
            assert accepted and result == value, (name, value, sdk, result)
    check('deep_gemm', expected, (2,2,0), True)
    check('deep_gemm', '1.1.0+v1.1.0.ppu2.2.0', (2,2,0), True)
    check('deep_gemm', expected, (2,2,1), False)
    check('deep_gemm', expected.replace('ppu2.2.0', 'ppu2.2.01'), (2,2,0), False)
    check('different_package', expected, (2,2,0), False)
    check('deep_gemm', expected+'malformed', (2,2,0), False)
    check('deep_gemm', expected, (0,0,0), False)
print(json.dumps(dict(case='source_metadata_version', cases=7, passed=True)))
