"""Apply the user's strict-order clarification before preregistration."""
from pathlib import Path
R=Path(__file__).resolve().parent
assert not (R/'PREREGISTRATION.json').exists()
p=R/'register_g1.py';s=p.read_text(encoding='utf-8')
assert "read('PILOT_ORDER_APPROVED.json')" in s
p.write_text(s.replace("read('PILOT_ORDER_APPROVED.json')","read('EXECUTION_ORDER_APPROVED.json')"),encoding='utf-8')
print('Strict-order approval reference updated before registration.')
