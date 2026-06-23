from robot.ccro_safety_gate import REQUIRED_CHECKS,evaluate_gate
def test_default_switch_is_blocked():
 r={k:False for k in REQUIRED_CHECKS}; assert not evaluate_gate("low_speed_switch",r,False,None,"OK").allowed
def test_missing_one_check_is_blocked():
 r={k:True for k in REQUIRED_CHECKS};r["self_collision_ok"]=False;assert not evaluate_gate("low_speed_switch",r,True,"OK","OK").allowed
def test_all_checks_and_explicit_phrase_allow_switch():
 r={k:True for k in REQUIRED_CHECKS};assert evaluate_gate("low_speed_switch",r,True,"OK","OK").allowed
