from risk.safety_policy import RiskLevel, SafetyDecision


class RobotCommandInterface:
    def send_velocity(self, joint_velocity):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError


class MockRobotCommand(RobotCommandInterface):
    def send_velocity(self, joint_velocity):
        print(f"mock velocity command: {joint_velocity}")

    def stop(self):
        print("mock stop command")

    def apply_safety_decision(self, decision: SafetyDecision):
        if decision.level == RiskLevel.STOP:
            self.stop()
        else:
            print(f"mock speed scale: {decision.speed_scale:.2f}")
