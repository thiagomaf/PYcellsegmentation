# src/status_manager.py
"""
Lightweight status tracking module for pipeline monitoring.
Kept separate from pipeline_utils to avoid heavy dependencies (cv2, numpy, etc.)
when used by the TUI.
"""
import os
import json
import logging
from datetime import datetime

from .file_paths import STATUS_DIR

logger = logging.getLogger(__name__)


class StatusManager:
    """
    Manages pipeline status tracking via a JSON file.
    Each config file gets its own status file in data/status/.
    """
    
    STEPS = ["Initialize", "Generate Jobs", "Segment", "Finalize"]
    
    def __init__(self, config_path: str):
        """
        Initialize the StatusManager for a specific config file.
        
        Args:
            config_path: Path to the configuration JSON file being processed.
        """
        self.config_name = os.path.basename(config_path)
        self.run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.status_file = os.path.join(
            STATUS_DIR, 
            f"pipeline_status_{os.path.splitext(self.config_name)[0]}.json"
        )
        self._ensure_status_dir()
        self._state = {
            "config_name": self.config_name,
            "config_path": config_path,
            "run_id": self.run_id,
            "status": "initializing",
            "current_step": self.STEPS[0],
            "current_step_index": 0,
            "progress": {
                "total": 0,
                "completed": 0,
                "failed": 0
            },
            "start_time": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "logs": []
        }
        self._write()
    
    def _ensure_status_dir(self):
        """Create the status directory if it doesn't exist."""
        if not os.path.exists(STATUS_DIR):
            try:
                os.makedirs(STATUS_DIR, exist_ok=True)
            except OSError as e:
                logger.error(f"Could not create status directory {STATUS_DIR}: {e}")
    
    def _write(self):
        """Write the current state to the status file."""
        self._state["last_updated"] = datetime.now().isoformat()
        try:
            with open(self.status_file, 'w') as f:
                json.dump(self._state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to write status file {self.status_file}: {e}")
    
    def set_step(self, step_name: str):
        """
        Set the current pipeline step.
        
        Args:
            step_name: One of "Initialize", "Generate Jobs", "Segment", "Finalize"
        """
        if step_name in self.STEPS:
            self._state["current_step"] = step_name
            self._state["current_step_index"] = self.STEPS.index(step_name)
            self._state["status"] = "running"
            self.log(f"Step started: {step_name}")
            self._write()
    
    def set_total_jobs(self, total: int):
        """Set the total number of jobs to process."""
        self._state["progress"]["total"] = total
        self.log(f"Total jobs to process: {total}")
        self._write()
    
    def job_completed(self, job_id: str, success: bool = True, message: str = ""):
        """
        Record a job completion.
        
        Args:
            job_id: Identifier for the completed job.
            success: Whether the job succeeded.
            message: Optional message about the job result.
        """
        if success:
            self._state["progress"]["completed"] += 1
            status_str = "SUCCESS"
        else:
            self._state["progress"]["failed"] += 1
            status_str = "FAILED"
        
        completed = self._state["progress"]["completed"]
        failed = self._state["progress"]["failed"]
        total = self._state["progress"]["total"]
        
        log_msg = f"[{completed + failed}/{total}] {job_id}: {status_str}"
        if message:
            log_msg += f" - {message}"
        self.log(log_msg)
        self._write()
    
    def log(self, message: str):
        """
        Add a log message to the status.
        
        Args:
            message: The log message to add.
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._state["logs"].append(f"[{timestamp}] {message}")
        # Keep only the last 100 log entries to prevent file bloat
        if len(self._state["logs"]) > 100:
            self._state["logs"] = self._state["logs"][-100:]
        self._write()
    
    def finish(self, success: bool = True, message: str = ""):
        """
        Mark the pipeline run as finished.
        
        Args:
            success: Whether the pipeline completed successfully.
            message: Optional final message.
        """
        self._state["status"] = "completed" if success else "error"
        self._state["current_step"] = "Finalize"
        self._state["current_step_index"] = len(self.STEPS) - 1
        self._state["end_time"] = datetime.now().isoformat()
        
        final_msg = "Pipeline completed successfully" if success else f"Pipeline failed: {message}"
        self.log(final_msg)
        self._write()
    
    @staticmethod
    def get_status_file_path(config_path: str) -> str:
        """Get the status file path for a given config without creating a StatusManager."""
        config_name = os.path.basename(config_path)
        return os.path.join(
            STATUS_DIR,
            f"pipeline_status_{os.path.splitext(config_name)[0]}.json"
        )
    
    @staticmethod
    def list_status_files() -> list:
        """List all status files in the status directory."""
        if not os.path.exists(STATUS_DIR):
            return []
        return [
            os.path.join(STATUS_DIR, f) 
            for f in os.listdir(STATUS_DIR) 
            if f.startswith("pipeline_status_") and f.endswith(".json")
        ]
    
    @staticmethod
    def read_status(status_file_path: str) -> dict | None:
        """Read and return the status from a status file."""
        if not os.path.exists(status_file_path):
            return None
        try:
            with open(status_file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read status file {status_file_path}: {e}")
            return None


__all__ = ["StatusManager"]

