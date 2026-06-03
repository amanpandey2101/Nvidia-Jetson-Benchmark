import time
import datetime
import logging

logger = logging.getLogger("benchmark.ucb")

class UCBEngine:
    def __init__(self, 
                 window_seconds: int = 85,
                 min_frames_in_window: int = 35,
                 max_score_threshold: float = 0.35,
                 last_score_threshold: float = 0.30,
                 alert_cooldown_seconds: int = 60,
                 ucb_required_fraction: float = 0.75,
                 child_confidence_threshold: float = 0.10,
                 adult_confidence_threshold: float = 0.10,
                 quiet_start_hour: int = 22,
                 quiet_end_hour: int = 6):
        
        self.window_seconds = window_seconds
        self.min_frames_in_window = min_frames_in_window
        self.max_score_threshold = max_score_threshold
        self.last_score_threshold = last_score_threshold
        self.alert_cooldown_seconds = alert_cooldown_seconds
        self.ucb_required_fraction = ucb_required_fraction
        self.child_confidence_threshold = child_confidence_threshold
        self.adult_confidence_threshold = adult_confidence_threshold
        self.quiet_start_hour = quiet_start_hour
        self.quiet_end_hour = quiet_end_hour
        
        # State
        self.window = []  # List of dicts: {"timestamp": float, "ucb_flag": bool, "ucb_score": float}
        self.last_alert_time = 0.0

    def in_quiet_hours(self, now_ts: float) -> bool:
        if self.quiet_start_hour == self.quiet_end_hour:
            return False
        
        dt = datetime.datetime.fromtimestamp(now_ts)
        h = dt.hour
        
        if self.quiet_start_hour < self.quiet_end_hour:
            return self.quiet_start_hour <= h < self.quiet_end_hour
        else:
            return h >= self.quiet_start_hour or h < self.quiet_end_hour

    def feed(self, detections: list, timestamp_override: float = 0.0) -> bool:
        """
        Feeds detections for a frame.
        detections format: list of dicts: {"class_id": int, "confidence": float, "box": [x1, y1, x2, y2]}
        Returns True if an alert is triggered, otherwise False.
        """
        now = timestamp_override if timestamp_override > 0.0 else time.time()
        
        child_detected = False
        adult_detected = False
        child_max_conf = 0.0
        
        # Evaluate raw frame detections
        for det in detections:
            class_id = det.get("class_id")
            confidence = det.get("confidence", 0.0)
            
            if class_id == 0:  # Child
                if confidence >= self.child_confidence_threshold:
                    child_detected = True
                    if confidence > child_max_conf:
                        child_max_conf = confidence
            elif class_id == 1:  # Adult
                if confidence >= self.adult_confidence_threshold:
                    adult_detected = True
        
        # Determine UCB status for current frame
        current_ucb_flag = child_detected and not adult_detected
        current_ucb_score = child_max_conf if current_ucb_flag else 0.0
        
        # Append to window
        self.window.append({
            "timestamp": now,
            "ucb_flag": current_ucb_flag,
            "ucb_score": current_ucb_score
        })
        
        # Evict old entries
        cutoff = now - self.window_seconds
        self.window = [entry for entry in self.window if entry["timestamp"] >= cutoff]
        
        n = len(self.window)
        max_score = max((entry["ucb_score"] for entry in self.window), default=0.0)
        last_score = self.window[-1]["ucb_score"] if self.window else 0.0
        
        # Count unattended positive frames
        unattended_count = sum(
            1 for entry in self.window 
            if entry["ucb_flag"] and entry["ucb_score"] >= self.last_score_threshold
        )
        
        # Check trigger criteria
        min_frames_ok = (n >= self.min_frames_in_window)
        max_score_ok = (max_score >= self.max_score_threshold)
        last_score_ok = (last_score >= self.last_score_threshold)
        
        required_positive = max(1, int(self.min_frames_in_window * self.ucb_required_fraction))
        duration_ok = (unattended_count >= required_positive)
        
        cooldown_elapsed = (now - self.last_alert_time) >= self.alert_cooldown_seconds
        quiet = self.in_quiet_hours(now)
        
        trigger = (
            min_frames_ok and
            duration_ok and
            max_score_ok and
            last_score_ok and
            cooldown_elapsed and
            not quiet
        )
        
        logger.debug(
            f"[ucb] window={n} unattended={unattended_count}/{required_positive} "
            f"max={max_score:.2f} last={last_score:.2f} child={int(child_detected)} "
            f"adult={int(adult_detected)} cooldown_ok={cooldown_elapsed} quiet={quiet}"
        )
        
        if trigger:
            self.last_alert_time = now
            logger.info(f"[ucb] *** ALERT TRIGGERED *** score={max_score:.2f} (Resetting window memory)")
            # Clean Slate: Wipe window memory so next alert requires a fresh sequence
            self.window.clear()
            return True
            
        return False
