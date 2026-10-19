"""Real-core stability gate; these diagnostic episodes are never study data."""
from pathlib import Path
import json
import random
from experiment.collect import episode


def main():
    config = json.loads(Path('configs/full.json').read_text())
    config['rounds'] = 40
    root, out = Path('.lab/open5gs').resolve(), Path('regression').resolve()
    for i, role in enumerate(['UDR', 'UDM', 'SMF', 'PCF']):
        result = episode(root, out, 'regression', 'benign_delay', i,
                         random.Random(67), config, target_override=role)
        log = (out/'raw'/result['episode']/'open5gs.log').read_text(errors='replace')
        if 'nghttp2_session_get_stream_remote_close() failed' in log:
            raise RuntimeError('KIE reply re-entered the HTTP/2 receive callback')
        if log.count('SUCCESS') < config['rounds']:
            raise RuntimeError('Not all registration rounds completed')


if __name__ == '__main__':
    main()
