| mode | allowed | reason | failed checks |
| --- | --- | --- | --- |
| dry_run | True | no trajectory-switch command is permitted | self_collision_ok, watchdog_ok, communication_ok, timestamp_ok, switch_state_error_ok, emergency_stop_ok, manufacturer_limits_verified |
| shadow | True | no trajectory-switch command is permitted | self_collision_ok, watchdog_ok, communication_ok, timestamp_ok, switch_state_error_ok, emergency_stop_ok, manufacturer_limits_verified |
| safety_layer | False | allow_real_robot_commands is false | self_collision_ok, watchdog_ok, communication_ok, timestamp_ok, switch_state_error_ok, emergency_stop_ok, manufacturer_limits_verified |
| low_speed_switch | False | allow_real_robot_commands is false | self_collision_ok, watchdog_ok, communication_ok, timestamp_ok, switch_state_error_ok, emergency_stop_ok, manufacturer_limits_verified |
