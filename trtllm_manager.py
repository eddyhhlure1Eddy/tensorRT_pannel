#!/usr/bin/env python3
"""
TensorRT-LLM Service Manager
A PyQt6-based GUI for managing TensorRT-LLM inference services with memory optimization.
Author: eddy
"""

import sys
import os
import json
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox,
    QComboBox, QTextEdit, QFileDialog, QTabWidget, QFormLayout,
    QCheckBox, QMessageBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QStatusBar, QSlider, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QFont, QPalette, QColor


# =============================================================================
# Configuration Management
# =============================================================================

CONFIG_FILE = Path(__file__).parent / "config.json"


@dataclass
class KvCacheSettings:
    """KV Cache configuration - optimized for low memory usage."""
    # TensorRT-LLM default is 0.9 which is INSANE - wastes 90% of free VRAM
    # We use 0.05 (5%) as sensible default, combined with max_tokens for precise control
    free_gpu_memory_fraction: float = 0.05
    max_tokens: int = 65536  # Enough for most use cases, ~10-15GB for large models
    enable_block_reuse: bool = True
    dtype: str = "auto"  # auto, fp8
    tokens_per_block: int = 32
    host_cache_size: int = 0  # 0 = no offloading


@dataclass
class ModelSettings:
    """Model and parallelism configuration."""
    model_path: str = "/mnt/e/fp4_models"
    venv_path: str = "/mnt/e/TensorRT-LLM/venv"
    trtllm_path: str = "/mnt/e/TensorRT-LLM"
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 3  # Default 3 GPUs for WSL2 multi-GPU
    moe_expert_parallel_size: int = 0  # 0 = auto, for MoE models
    moe_tensor_parallel_size: int = 0  # 0 = auto, for MoE models
    cuda_visible_devices: str = ""  # empty = all GPUs
    dtype: str = "auto"  # auto for quantized models (NVFP4, FP8), bfloat16/float16 for unquantized


@dataclass
class InferenceSettings:
    """Inference configuration."""
    max_seq_len: int = 4096
    max_batch_size: int = 8
    max_num_tokens: int = 0  # 0 = auto (same as max_seq_len)
    enable_chunked_prefill: bool = False
    disable_overlap_scheduler: bool = True  # True = reduce CPU usage, False = max throughput
    disable_autotuner: bool = False  # Disable MoE GEMM autotuner to avoid profiling OOM
    cuda_graph_enabled: bool = True  # CUDA graph reduces kernel launch overhead but may cause warmup issues


@dataclass
class ApiSettings:
    """API server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    enable_port_forward: bool = True  # Auto-setup WSL2 port forwarding


@dataclass
class WslSettings:
    """WSL configuration."""
    distro: str = "Ubuntu-24.04"
    cpu_limit: int = 0  # 0 = no limit, N = limit to N processors
    memory_limit: int = 0  # 0 = no limit, N = limit to N GB


@dataclass
class AppConfig:
    """Complete application configuration."""
    wsl: WslSettings = field(default_factory=WslSettings)
    model: ModelSettings = field(default_factory=ModelSettings)
    kv_cache: KvCacheSettings = field(default_factory=KvCacheSettings)
    inference: InferenceSettings = field(default_factory=InferenceSettings)
    api: ApiSettings = field(default_factory=ApiSettings)


class ConfigManager:
    """Manages application configuration persistence."""

    def __init__(self, config_path: Path = CONFIG_FILE):
        self.config_path = config_path
        self.config = self.load()

    def load(self) -> AppConfig:
        """Load configuration from file or return defaults."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return self._dict_to_config(data)
            except Exception as e:
                print(f"Failed to load config: {e}")
        return AppConfig()

    def save(self) -> None:
        """Save current configuration to file."""
        data = self._config_to_dict(self.config)
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _config_to_dict(self, config: AppConfig) -> Dict[str, Any]:
        """Convert AppConfig to dictionary."""
        return {
            'wsl': asdict(config.wsl),
            'model': asdict(config.model),
            'kv_cache': asdict(config.kv_cache),
            'inference': asdict(config.inference),
            'api': asdict(config.api),
        }

    def _dict_to_config(self, data: Dict[str, Any]) -> AppConfig:
        """Convert dictionary to AppConfig."""
        return AppConfig(
            wsl=WslSettings(**data.get('wsl', {})),
            model=ModelSettings(**data.get('model', {})),
            kv_cache=KvCacheSettings(**data.get('kv_cache', {})),
            inference=InferenceSettings(**data.get('inference', {})),
            api=ApiSettings(**data.get('api', {})),
        )


# =============================================================================
# Script Generator
# =============================================================================

class ScriptGenerator:
    """Generates TensorRT-LLM startup scripts with proper parameter passing."""

    TEMPLATE = '''#!/usr/bin/env python3
"""
Auto-generated TensorRT-LLM startup script
Generated by TensorRT-LLM Service Manager
Author: eddy
"""
import os
import sys
import asyncio
import logging

# {autotuner_env}

# Configure logging to both stdout and file
os.environ["TLLM_LOG_LEVEL"] = "INFO"

# Create log file in script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, "trtllm_service.log")

# Setup dual logging: stdout + file
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Clear any existing handlers
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# File handler - append mode for persistence
file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
))
logger.addHandler(file_handler)

# Stdout handler for panel display
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
stdout_handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
))
logger.addHandler(stdout_handler)

# Log startup marker
import datetime as _dt
logger.info("=" * 80)
logger.info("TensorRT-LLM Service Log - Started at " + str(_dt.datetime.now()))
logger.info("Log file: " + log_file)
logger.info("=" * 80)

logger = logging.getLogger(__name__)

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from tensorrt_llm import LLM
from tensorrt_llm.llmapi import KvCacheConfig, CudaGraphConfig
from tensorrt_llm.serve import OpenAIServer


def main():
    logger.info("=" * 60)
    logger.info("TensorRT-LLM Service Manager - Starting")
    logger.info("=" * 60)

    # KV Cache Configuration - Memory Optimized
    # free_gpu_memory_fraction=0 means use max_tokens only, no percentage-based allocation
    logger.info("Configuring KV Cache...")
    logger.info("  free_gpu_memory_fraction: {free_gpu_memory_fraction}")
    logger.info("  max_tokens: {max_tokens}")
    logger.info("  enable_block_reuse: {enable_block_reuse}")

    kv_cache_config = KvCacheConfig(
        free_gpu_memory_fraction={free_gpu_memory_fraction},
        max_tokens={max_tokens},
        enable_block_reuse={enable_block_reuse},
        dtype="{kv_cache_dtype}",
        tokens_per_block={tokens_per_block},
        host_cache_size={host_cache_size},
    )

    # CUDA Graph Configuration - Reduces CPU kernel launch overhead
    # {cuda_graph_comment}
    cuda_graph_config = {cuda_graph_config}

    # Model Configuration
    logger.info("Loading model...")
    logger.info("  model_path: {model_path}")
    logger.info("  tensor_parallel_size: {tensor_parallel_size}")
    logger.info("  pipeline_parallel_size: {pipeline_parallel_size}")
    logger.info("  moe_expert_parallel_size: {moe_expert_parallel_size_log}")
    logger.info("  moe_tensor_parallel_size: {moe_tensor_parallel_size_log}")
    logger.info("  max_seq_len: {max_seq_len}")
    logger.info("  max_batch_size: {max_batch_size}")
    logger.info("  max_num_tokens: {max_num_tokens}")
    logger.info("  enable_autotuner: {enable_autotuner}")
    logger.info("  cuda_graph_enabled: {cuda_graph_enabled}")

    llm = LLM(
        model="{model_path}",
        tensor_parallel_size={tensor_parallel_size},
        pipeline_parallel_size={pipeline_parallel_size},
{moe_params}        kv_cache_config=kv_cache_config,
        cuda_graph_config=cuda_graph_config,
        max_seq_len={max_seq_len},
        max_batch_size={max_batch_size},
        max_num_tokens={max_num_tokens},
        enable_chunked_prefill={enable_chunked_prefill},
        disable_overlap_scheduler={disable_overlap_scheduler},
        enable_autotuner={enable_autotuner},
        dtype="{dtype}",
    )

    logger.info("=" * 60)
    logger.info("Model loaded successfully!")
    logger.info("Starting OpenAI-compatible API server...")
    logger.info("API URL: http://{api_host}:{api_port}")
    logger.info("=" * 60)

    # Start OpenAI-compatible API server
    # tool_parser, server_role, metadata_server_cfg are required but can be None
    server = OpenAIServer(
        llm=llm,
        model="{model_path}",
        tool_parser=None,
        server_role=None,
        metadata_server_cfg=None,
    )
    asyncio.run(server("{api_host}", {api_port}))


if __name__ == "__main__":
    main()
'''

    @classmethod
    def generate(cls, config: AppConfig) -> str:
        """Generate startup script from configuration."""
        # Auto-calculate max_tokens if not manually set or if it's the default
        max_tokens = config.kv_cache.max_tokens
        calculated = config.inference.max_seq_len * config.inference.max_batch_size
        if max_tokens == 32768:  # Default value, use calculated
            max_tokens = calculated

        # max_num_tokens: 0 = auto (use max_seq_len)
        max_num_tokens = config.inference.max_num_tokens
        if max_num_tokens == 0:
            max_num_tokens = config.inference.max_seq_len

        # MoE parallel sizes: 0 = auto, only pass parameters when explicitly set
        # CRITICAL: Passing None causes "moe_tp * moe_ep must equal moe_world_size" error
        # Solution: Don't pass the parameter at all when it should be auto-detected
        moe_params_lines = []
        moe_ep_val = config.model.moe_expert_parallel_size
        moe_tp_val = config.model.moe_tensor_parallel_size

        if moe_ep_val > 0:
            moe_params_lines.append(f"        moe_expert_parallel_size={moe_ep_val},")
        if moe_tp_val > 0:
            moe_params_lines.append(f"        moe_tensor_parallel_size={moe_tp_val},")

        moe_params = "\n".join(moe_params_lines) + "\n" if moe_params_lines else ""
        moe_ep_log = moe_ep_val if moe_ep_val > 0 else "auto"
        moe_tp_log = moe_tp_val if moe_tp_val > 0 else "auto"

        # Autotuner configuration
        # CRITICAL: enable_autotuner must be passed to LLM() constructor, NOT env var!
        # Environment variable TRTLLM_DISABLE_AUTOTUNER does NOT work reliably
        if config.inference.disable_autotuner:
            autotuner_env = "# Autotuner DISABLED via LLM(enable_autotuner=False)"
            enable_autotuner = "False"
        else:
            autotuner_env = "# Autotuner enabled (default)"
            enable_autotuner = "True"

        # CUDA Graph configuration
        # Disabling CUDA graphs can help with multi-GPU warmup issues
        if config.inference.cuda_graph_enabled:
            cuda_graph_comment = "CUDA graphs ENABLED for reduced kernel launch overhead"
            cuda_graph_config = f"CudaGraphConfig(max_batch_size={config.inference.max_batch_size}, enable_padding=True)"
        else:
            cuda_graph_comment = "CUDA graphs DISABLED to avoid warmup issues on multi-GPU"
            cuda_graph_config = "None"

        return cls.TEMPLATE.format(
            autotuner_env=autotuner_env,
            free_gpu_memory_fraction=config.kv_cache.free_gpu_memory_fraction,
            max_tokens=max_tokens,
            enable_block_reuse=config.kv_cache.enable_block_reuse,
            kv_cache_dtype=config.kv_cache.dtype,
            tokens_per_block=config.kv_cache.tokens_per_block,
            host_cache_size=config.kv_cache.host_cache_size,
            model_path=config.model.model_path,
            tensor_parallel_size=config.model.tensor_parallel_size,
            pipeline_parallel_size=config.model.pipeline_parallel_size,
            moe_params=moe_params,
            moe_expert_parallel_size_log=moe_ep_log,
            moe_tensor_parallel_size_log=moe_tp_log,
            max_seq_len=config.inference.max_seq_len,
            max_batch_size=config.inference.max_batch_size,
            max_num_tokens=max_num_tokens,
            enable_chunked_prefill=config.inference.enable_chunked_prefill,
            disable_overlap_scheduler=config.inference.disable_overlap_scheduler,
            enable_autotuner=enable_autotuner,
            cuda_graph_comment=cuda_graph_comment,
            cuda_graph_config=cuda_graph_config,
            cuda_graph_enabled=config.inference.cuda_graph_enabled,
            dtype=config.model.dtype,
            api_host=config.api.host,
            api_port=config.api.port,
        )


# =============================================================================
# Path Utilities
# =============================================================================

def windows_to_wsl_path(win_path: str) -> str:
    """Convert Windows path to WSL path."""
    if not win_path:
        return ""
    path = win_path.replace('\\', '/')
    if len(path) >= 2 and path[1] == ':':
        drive = path[0].lower()
        path = f"/mnt/{drive}{path[2:]}"
    return path


def wsl_to_windows_path(wsl_path: str) -> str:
    """Convert WSL path to Windows path."""
    if not wsl_path:
        return ""
    if wsl_path.startswith('/mnt/'):
        parts = wsl_path[5:].split('/', 1)
        if parts:
            drive = parts[0].upper()
            rest = parts[1] if len(parts) > 1 else ""
            return f"{drive}:\\{rest.replace('/', '\\')}"
    return wsl_path


# =============================================================================
# Background Threads
# =============================================================================

class LogReaderThread(QThread):
    """Thread for reading process output."""
    log_received = pyqtSignal(str)

    def __init__(self, process: subprocess.Popen):
        super().__init__()
        self.process = process
        self.running = True

    def run(self):
        try:
            while self.running:
                if self.process.poll() is not None:
                    for line in self.process.stdout:
                        if line:
                            self.log_received.emit(line.strip())
                    break
                line = self.process.stdout.readline()
                if line:
                    self.log_received.emit(line.strip())
        except Exception as e:
            self.log_received.emit(f"[ERROR] Log reader: {e}")

    def stop(self):
        self.running = False


class GPUMonitorThread(QThread):
    """Thread for monitoring GPU status."""
    gpu_updated = pyqtSignal(list)

    def __init__(self, wsl_distro: str = "Ubuntu-24.04"):
        super().__init__()
        self.running = True
        self.wsl_distro = wsl_distro

    def run(self):
        while self.running:
            try:
                result = subprocess.run(
                    ["wsl", "-d", self.wsl_distro, "bash", "-c",
                     "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits"],
                    capture_output=True, timeout=5, encoding='utf-8', errors='ignore'
                )
                if result.returncode == 0:
                    gpus = []
                    for line in result.stdout.strip().split('\n'):
                        if line:
                            parts = [p.strip() for p in line.split(',')]
                            if len(parts) >= 6:
                                gpus.append({
                                    'index': parts[0],
                                    'name': parts[1],
                                    'memory_used': parts[2],
                                    'memory_total': parts[3],
                                    'utilization': parts[4],
                                    'temperature': parts[5]
                                })
                    self.gpu_updated.emit(gpus)
            except Exception:
                pass
            self.msleep(2000)

    def stop(self):
        self.running = False


# =============================================================================
# Process Manager
# =============================================================================

class ProcessManager:
    """Manages TensorRT-LLM service process lifecycle."""

    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.log_thread: Optional[LogReaderThread] = None
        self.run_id: Optional[str] = None
        self.pidfile: Optional[str] = None
        self.script_path: Optional[Path] = None

    @property
    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self, config: AppConfig, log_callback) -> bool:
        """Start the TensorRT-LLM service."""
        if self.is_running:
            return False

        # Generate startup script
        self.run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        script_content = ScriptGenerator.generate(config)
        self.script_path = Path(__file__).parent / f"startup_script_{self.run_id}.py"

        with open(self.script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)

        wsl_script = windows_to_wsl_path(str(self.script_path))
        python_path = f"{config.model.venv_path}/bin/python3"
        self.pidfile = f"/tmp/trtllm_manager_{self.run_id}.pid"

        # Build environment setup for CUDA_VISIBLE_DEVICES and NCCL (WSL2 multi-GPU fix)
        env_setup = ""
        if config.model.cuda_visible_devices:
            env_setup = f"export CUDA_VISIBLE_DEVICES={config.model.cuda_visible_devices} && "

        # Add NCCL environment variables for WSL2 multi-GPU support
        # NCCL_CUMEM_ENABLE=0 is the KEY FIX for WSL2 NCCL "cuda error 999"
        # See: https://github.com/NVIDIA/nccl/issues/1653
        total_gpus = config.model.tensor_parallel_size * config.model.pipeline_parallel_size
        moe_ep = config.model.moe_expert_parallel_size
        needs_multi_gpu = total_gpus > 1 or moe_ep > 1
        if needs_multi_gpu:
            # NCCL_CUMEM_ENABLE=0: Fix WSL2 NCCL "cuda error 999"
            # NCCL_BLOCKING_WAIT=1: Use blocking wait instead of busy-polling (reduces CPU usage)
            nccl_env = "export NCCL_CUMEM_ENABLE=0 && export NCCL_BLOCKING_WAIT=1 && "
            env_setup = nccl_env + env_setup

        # Add TRTLLM_DISABLE_AUTOTUNER for MoE GEMM profiling OOM fix
        # CRITICAL: Must be set in bash command line, NOT in Python script!
        # MPI spawns child processes that won't inherit os.environ from Python runtime
        if config.inference.disable_autotuner:
            env_setup = "export TRTLLM_DISABLE_AUTOTUNER=1 && " + env_setup

        # Build command
        bash_cmd = (
            f"set -m; export PYTHONUNBUFFERED=1; {env_setup}"
            f"{python_path} {wsl_script} & echo $! > '{self.pidfile}'; wait $!"
        )
        cmd = ["wsl", "-d", config.wsl.distro, "bash", "-lc", bash_cmd]

        try:
            self.process = subprocess.Popen(
                cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, encoding='utf-8', errors='ignore'
            )
            self.log_thread = LogReaderThread(self.process)
            self.log_thread.log_received.connect(log_callback)
            self.log_thread.start()
            return True
        except Exception as e:
            log_callback(f"[ERROR] Failed to start: {e}")
            self.process = None
            return False

    def stop(self, wsl_distro: str) -> None:
        """Stop the TensorRT-LLM service."""
        if not self.process:
            return

        # Kill via pidfile
        if self.pidfile:
            kill_cmd = (
                f"if [ -f '{self.pidfile}' ]; then "
                f"pid=$(cat '{self.pidfile}'); "
                f"kill -15 -- -$pid 2>/dev/null || true; "
                f"sleep 1; "
                f"kill -9 -- -$pid 2>/dev/null || true; "
                f"rm -f '{self.pidfile}'; "
                f"fi"
            )
            try:
                subprocess.run(
                    ["wsl", "-d", wsl_distro, "bash", "-lc", kill_cmd],
                    capture_output=True, timeout=5
                )
            except Exception:
                pass

        # Stop log thread
        if self.log_thread:
            self.log_thread.stop()
            self.log_thread.wait(2000)

        # Terminate process
        try:
            if self.process:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=3)
        except Exception:
            pass

        # Cleanup
        self.process = None
        self.log_thread = None
        self.run_id = None
        self.pidfile = None

    def force_kill(self, wsl_distro: str, api_port: int) -> None:
        """Force kill all related processes."""
        # Kill commands inside WSL - more comprehensive patterns
        kill_commands = [
            # Kill by pidfile
            "for f in /tmp/trtllm_manager_*.pid; do [ -f \"$f\" ] && pid=$(cat \"$f\") && kill -9 -$pid 2>/dev/null; kill -9 $pid 2>/dev/null; rm -f \"$f\"; done",
            # Kill specific processes by name pattern
            "pkill -9 -f 'startup_script_' 2>/dev/null || true",
            "pkill -9 -f 'trtllm-serve' 2>/dev/null || true",
            "pkill -9 -f 'tensorrt_llm' 2>/dev/null || true",
            "pkill -9 -f 'mpirun' 2>/dev/null || true",
            "pkill -9 -f 'OpenAIServer' 2>/dev/null || true",
            # Kill port usage
            f"fuser -k {api_port}/tcp 2>/dev/null || true",
            # Kill all python processes in venv (aggressive)
            "pkill -9 -f '/TensorRT-LLM/venv/bin/python' 2>/dev/null || true",
        ]

        for cmd in kill_commands:
            try:
                subprocess.run(
                    ["wsl", "-d", wsl_distro, "bash", "-c", cmd],
                    capture_output=True, timeout=10
                )
            except Exception:
                pass

        if self.process:
            try:
                self.process.kill()
            except Exception:
                pass

        if self.log_thread:
            self.log_thread.stop()

        self.process = None
        self.log_thread = None
        self.run_id = None
        self.pidfile = None

    def terminate_wsl(self, wsl_distro: str) -> bool:
        """Terminate entire WSL distro - nuclear option."""
        try:
            result = subprocess.run(
                ["wsl", "--terminate", wsl_distro],
                capture_output=True, timeout=30, encoding='gbk', errors='replace'
            )
            self.process = None
            self.log_thread = None
            self.run_id = None
            self.pidfile = None
            return result.returncode == 0
        except Exception:
            return False


# =============================================================================
# UI Components
# =============================================================================

class ServiceControlTab(QWidget):
    """Service control and log display tab."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Control buttons
        ctrl_group = QGroupBox("Service Control")
        ctrl_layout = QHBoxLayout(ctrl_group)

        self.btn_start = QPushButton("Start Service")
        self.btn_start.setMinimumHeight(50)
        self.btn_start.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")

        self.btn_stop = QPushButton("Stop Service")
        self.btn_stop.setMinimumHeight(50)
        self.btn_stop.setStyleSheet("background-color: #f44336; color: white; font-weight: bold;")
        self.btn_stop.setEnabled(False)

        self.btn_restart = QPushButton("Restart Service")
        self.btn_restart.setMinimumHeight(50)
        self.btn_restart.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")

        self.btn_force_kill = QPushButton("Force Kill")
        self.btn_force_kill.setMinimumHeight(50)
        self.btn_force_kill.setStyleSheet("background-color: #9C27B0; color: white; font-weight: bold;")
        self.btn_force_kill.setToolTip("Execute wsl --terminate to kill all processes")

        self.btn_save = QPushButton("Save Config")
        self.btn_save.setMinimumHeight(50)

        ctrl_layout.addWidget(self.btn_start)
        ctrl_layout.addWidget(self.btn_stop)
        ctrl_layout.addWidget(self.btn_restart)
        ctrl_layout.addWidget(self.btn_force_kill)
        ctrl_layout.addWidget(self.btn_save)
        layout.addWidget(ctrl_group)

        # Status display
        status_group = QGroupBox("Service Status")
        status_layout = QFormLayout(status_group)

        self.lbl_status = QLabel("Stopped")
        self.lbl_status.setStyleSheet("color: red; font-weight: bold; font-size: 14px;")
        self.lbl_pid = QLabel("-")
        self.lbl_api_url = QLabel("-")

        status_layout.addRow("Status:", self.lbl_status)
        status_layout.addRow("Process ID:", self.lbl_pid)
        status_layout.addRow("API URL:", self.lbl_api_url)
        layout.addWidget(status_group)

        # Log display
        log_group = QGroupBox("Service Log")
        log_layout = QVBoxLayout(log_group)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        self.log_text.setStyleSheet("background-color: #1e1e1e; color: #dcdcdc;")

        btn_layout = QHBoxLayout()
        btn_clear = QPushButton("Clear Log")
        btn_clear.clicked.connect(lambda: self.log_text.clear())
        btn_layout.addWidget(btn_clear)
        btn_layout.addStretch()

        log_layout.addWidget(self.log_text)
        log_layout.addLayout(btn_layout)
        layout.addWidget(log_group, stretch=1)

    def set_running(self, running: bool, pid: str = "-", api_url: str = "-"):
        """Update UI state based on service status."""
        self.btn_start.setEnabled(not running)
        self.btn_stop.setEnabled(running)
        self.lbl_status.setText("Running" if running else "Stopped")
        self.lbl_status.setStyleSheet(
            f"color: {'green' if running else 'red'}; font-weight: bold; font-size: 14px;"
        )
        self.lbl_pid.setText(pid)
        self.lbl_api_url.setText(api_url)

    def append_log(self, msg: str):
        """Append message to log display."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {msg}")
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())


class ModelConfigTab(QWidget):
    """Model and parallelism configuration tab."""

    def __init__(self, config: AppConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # WSL settings
        wsl_group = QGroupBox("WSL Configuration")
        wsl_layout = QFormLayout(wsl_group)

        self.combo_wsl_distro = QComboBox()
        self.combo_wsl_distro.setEditable(True)
        self.combo_wsl_distro.addItems(["Ubuntu-24.04", "Ubuntu-22.04", "Ubuntu", "Debian"])
        self.combo_wsl_distro.setCurrentText(self.config.wsl.distro)
        wsl_layout.addRow("WSL Distro:", self.combo_wsl_distro)

        # CPU limit for WSL2
        self.spin_wsl_cpu = QSpinBox()
        self.spin_wsl_cpu.setRange(0, 64)
        self.spin_wsl_cpu.setValue(self.config.wsl.cpu_limit)
        self.spin_wsl_cpu.setToolTip(
            "Limit WSL2 CPU processors.\n"
            "0 = No limit (use all CPUs)\n"
            "6 = ~20% for i9-14900K (32 threads)\n"
            "Requires WSL restart to take effect."
        )
        wsl_layout.addRow("CPU Limit (0=all):", self.spin_wsl_cpu)

        # Memory limit for WSL2
        self.spin_wsl_memory = QSpinBox()
        self.spin_wsl_memory.setRange(0, 512)
        self.spin_wsl_memory.setValue(self.config.wsl.memory_limit)
        self.spin_wsl_memory.setSuffix(" GB")
        self.spin_wsl_memory.setToolTip(
            "Limit WSL2 memory usage.\n"
            "0 = No limit (use default ~50% of RAM)\n"
            "Requires WSL restart to take effect."
        )
        wsl_layout.addRow("Memory Limit (0=auto):", self.spin_wsl_memory)

        # Apply and restart buttons
        wsl_btn_layout = QHBoxLayout()
        self.btn_apply_wsl = QPushButton("Apply .wslconfig")
        self.btn_apply_wsl.clicked.connect(self._apply_wslconfig)
        self.btn_apply_wsl.setToolTip("Write settings to .wslconfig file")

        self.btn_restart_wsl = QPushButton("Restart WSL")
        self.btn_restart_wsl.clicked.connect(self._restart_wsl)
        self.btn_restart_wsl.setStyleSheet("background-color: #FF9800; color: white;")
        self.btn_restart_wsl.setToolTip("Run 'wsl --shutdown' to apply new settings")

        wsl_btn_layout.addWidget(self.btn_apply_wsl)
        wsl_btn_layout.addWidget(self.btn_restart_wsl)
        wsl_layout.addRow("", wsl_btn_layout)

        layout.addWidget(wsl_group)

        # Path settings
        path_group = QGroupBox("Path Settings")
        path_layout = QFormLayout(path_group)

        self.edit_model_path = QLineEdit(self.config.model.model_path)
        self.edit_model_path.setPlaceholderText("WSL path to model directory")
        btn_model = QPushButton("Browse...")
        btn_model.clicked.connect(lambda: self._browse_path(self.edit_model_path))
        model_row = QHBoxLayout()
        model_row.addWidget(self.edit_model_path)
        model_row.addWidget(btn_model)

        self.edit_venv_path = QLineEdit(self.config.model.venv_path)
        self.edit_venv_path.setPlaceholderText("WSL path to Python venv")
        btn_venv = QPushButton("Browse...")
        btn_venv.clicked.connect(lambda: self._browse_path(self.edit_venv_path))
        venv_row = QHBoxLayout()
        venv_row.addWidget(self.edit_venv_path)
        venv_row.addWidget(btn_venv)

        path_layout.addRow("Model Path:", model_row)
        path_layout.addRow("Venv Path:", venv_row)
        layout.addWidget(path_group)

        # Parallelism settings
        parallel_group = QGroupBox("Parallelism Settings")
        parallel_layout = QFormLayout(parallel_group)

        # Detect model config button
        detect_layout = QHBoxLayout()
        self.btn_detect_model = QPushButton("Detect Model Config")
        self.btn_detect_model.clicked.connect(self._detect_model_config)
        self.lbl_model_info = QLabel("Click to detect valid TP values")
        self.lbl_model_info.setStyleSheet("color: #888;")
        detect_layout.addWidget(self.btn_detect_model)
        detect_layout.addWidget(self.lbl_model_info, 1)
        parallel_layout.addRow("", detect_layout)

        self.spin_tp = QSpinBox()
        self.spin_tp.setRange(1, 8)
        self.spin_tp.setValue(self.config.model.tensor_parallel_size)
        self.spin_tp.setToolTip(
            "Tensor Parallel: split model across GPUs.\n"
            "Must divide num_attention_heads evenly!\n"
            "Click 'Detect Model Config' to see valid values."
        )

        self.spin_pp = QSpinBox()
        self.spin_pp.setRange(1, 8)
        self.spin_pp.setValue(self.config.model.pipeline_parallel_size)
        self.spin_pp.setToolTip(
            "Pipeline Parallel: split model layers across GPUs.\n"
            "RECOMMENDED for multi-GPU on WSL2!\n"
            "Example: 3 GPUs with 48 layers -> PP=3 (16 layers per GPU)\n"
            "Works with any model, no attention head constraints."
        )

        # MoE Expert Parallel (for MoE models like Qwen3-MoE, DeepSeek)
        self.spin_moe_ep = QSpinBox()
        self.spin_moe_ep.setRange(0, 8)
        self.spin_moe_ep.setValue(self.config.model.moe_expert_parallel_size)
        self.spin_moe_ep.setToolTip(
            "MoE Expert Parallel: distribute experts across GPUs.\n"
            "0 = Auto (RECOMMENDED - let TensorRT-LLM decide)\n"
            "NOTE: MoE EP requires world_size (TP*PP) >= EP value.\n"
            "For most cases, just use PP=N and leave EP=0."
        )

        # MoE Tensor Parallel (for MoE models)
        self.spin_moe_tp = QSpinBox()
        self.spin_moe_tp.setRange(0, 8)
        self.spin_moe_tp.setValue(self.config.model.moe_tensor_parallel_size)
        self.spin_moe_tp.setToolTip(
            "MoE Tensor Parallel: split expert weights across GPUs.\n"
            "0 = Auto (let TensorRT-LLM decide)\n"
            "Usually keep at 0 or 1 when using EP."
        )

        self.edit_cuda_devices = QLineEdit(self.config.model.cuda_visible_devices)
        self.edit_cuda_devices.setPlaceholderText("Empty = all GPUs, e.g. 0,1,2")
        self.edit_cuda_devices.setToolTip(
            "CUDA_VISIBLE_DEVICES: which GPUs to use.\n"
            "Empty = all GPUs.\n"
            "Note: TP * PP must <= number of visible GPUs!"
        )

        self.combo_dtype = QComboBox()
        self.combo_dtype.addItems(["auto", "bfloat16", "float16"])
        self.combo_dtype.setToolTip(
            "auto: Let TensorRT-LLM detect dtype from model config (REQUIRED for quantized models like NVFP4/FP8)\n"
            "bfloat16: Force BF16 precision (only for unquantized models)\n"
            "float16: Force FP16 precision (only for unquantized models)"
        )
        self.combo_dtype.setCurrentText(self.config.model.dtype)

        parallel_layout.addRow("Tensor Parallel (TP):", self.spin_tp)
        parallel_layout.addRow("Pipeline Parallel (PP):", self.spin_pp)
        parallel_layout.addRow("MoE Expert Parallel (EP):", self.spin_moe_ep)
        parallel_layout.addRow("MoE Tensor Parallel:", self.spin_moe_tp)
        parallel_layout.addRow("CUDA Devices:", self.edit_cuda_devices)
        parallel_layout.addRow("Model Dtype:", self.combo_dtype)
        layout.addWidget(parallel_group)

        layout.addStretch()

    def _detect_model_config(self):
        """Detect model config and show valid TP values."""
        model_path = self.edit_model_path.text()
        if not model_path:
            self.lbl_model_info.setText("Please set model path first!")
            self.lbl_model_info.setStyleSheet("color: #FF5722;")
            return

        # Convert WSL path to Windows path for reading
        if model_path.startswith("/mnt/"):
            # /mnt/e/xxx -> E:\xxx
            parts = model_path.split("/")
            if len(parts) >= 3:
                drive = parts[2].upper()
                rest = "\\".join(parts[3:])
                win_path = f"{drive}:\\{rest}"
            else:
                win_path = model_path
        else:
            win_path = model_path

        config_file = Path(win_path) / "config.json"
        if not config_file.exists():
            self.lbl_model_info.setText(f"config.json not found in {win_path}")
            self.lbl_model_info.setStyleSheet("color: #FF5722;")
            return

        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                model_config = json.load(f)

            num_heads = model_config.get("num_attention_heads", 0)
            num_experts = model_config.get("num_experts", 0)
            num_layers = model_config.get("num_hidden_layers", 0)

            # Calculate valid TP values (must divide num_heads evenly)
            valid_tp = [i for i in range(1, 9) if num_heads % i == 0]

            info_parts = [f"Heads={num_heads}", f"Layers={num_layers}"]
            if num_experts > 0:
                info_parts.append(f"Experts={num_experts}")
            info_parts.append(f"Valid TP: {valid_tp}")

            self.lbl_model_info.setText(" | ".join(info_parts))
            self.lbl_model_info.setStyleSheet("color: #4CAF50;")

            # Check current config and suggest best combo
            current_tp = self.spin_tp.value()
            current_pp = self.spin_pp.value()
            total_gpus = current_tp * current_pp

            # For MoE models, recommend using Expert Parallelism
            if num_experts > 0:
                moe_suggestion = (
                    f"MoE Model Detected!\n\n"
                    f"num_experts={num_experts}\n"
                    f"num_attention_heads={num_heads}\n"
                    f"num_hidden_layers={num_layers}\n\n"
                    f"RECOMMENDED for MoE models:\n"
                    f"  * Use Expert Parallel (EP) instead of TP/PP\n"
                    f"  * Set MoE Expert Parallel = number of GPUs\n"
                    f"  * Keep TP=1, PP=1\n\n"
                    f"Example for 3 GPUs:\n"
                    f"  TP=1, PP=1, MoE EP=3\n\n"
                    f"This distributes {num_experts} experts across GPUs\n"
                    f"without needing to evenly divide attention heads!"
                )
                QMessageBox.information(self, "MoE Model Configuration", moe_suggestion)
                return

            if current_tp not in valid_tp:
                # Build suggestion for alternative configs
                suggestion = (
                    f"Current TP={current_tp} is INVALID!\n\n"
                    f"num_attention_heads={num_heads}\n"
                    f"num_hidden_layers={num_layers}\n"
                    f"Valid TP values: {valid_tp}\n\n"
                )

                # Find best TP/PP combos for desired GPU count
                if total_gpus > 1:
                    suggestion += f"RECOMMENDATIONS for {total_gpus} GPUs:\n"
                    found_combo = False
                    for tp in valid_tp:
                        if total_gpus % tp == 0:
                            pp = total_gpus // tp
                            if pp <= num_layers:
                                layers_per_gpu = num_layers // pp
                                extra = num_layers % pp
                                dist = f"{layers_per_gpu} layers/GPU"
                                if extra:
                                    dist += f" (+1 on {extra} GPUs)"
                                suggestion += f"  * TP={tp}, PP={pp} -> {dist}\n"
                                found_combo = True

                    if not found_combo:
                        suggestion += f"  No valid combo. Try TP=1, PP={total_gpus}\n"
                else:
                    suggestion += "Set TP=1 for single GPU.\n"

                suggestion += f"\nPP (Pipeline Parallel) does NOT require even division!"

                QMessageBox.warning(self, "Invalid Configuration", suggestion)

        except Exception as e:
            self.lbl_model_info.setText(f"Error reading config: {e}")
            self.lbl_model_info.setStyleSheet("color: #FF5722;")

    def _browse_path(self, line_edit: QLineEdit):
        path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if path:
            wsl_path = windows_to_wsl_path(path)
            line_edit.setText(wsl_path)

    def _apply_wslconfig(self):
        """Write WSL2 resource limits to .wslconfig file."""
        cpu_limit = self.spin_wsl_cpu.value()
        memory_limit = self.spin_wsl_memory.value()

        # Build .wslconfig content
        lines = ["[wsl2]"]
        if cpu_limit > 0:
            lines.append(f"processors={cpu_limit}")
        if memory_limit > 0:
            lines.append(f"memory={memory_limit}GB")

        # Add some reasonable defaults
        lines.append("swap=0")
        lines.append("localhostForwarding=true")

        config_content = "\n".join(lines) + "\n"

        # Write to user's home directory
        wslconfig_path = Path.home() / ".wslconfig"
        try:
            with open(wslconfig_path, 'w', encoding='utf-8') as f:
                f.write(config_content)
            QMessageBox.information(
                self, "Success",
                f"Saved to {wslconfig_path}\n\n"
                f"Content:\n{config_content}\n"
                "Click 'Restart WSL' to apply changes."
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to write .wslconfig: {e}")

    def _restart_wsl(self):
        """Restart WSL to apply .wslconfig changes."""
        reply = QMessageBox.question(
            self, "Restart WSL",
            "This will run 'wsl --shutdown' to restart WSL.\n"
            "All WSL processes will be terminated!\n\n"
            "Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            try:
                result = subprocess.run(
                    ["wsl", "--shutdown"],
                    capture_output=True, timeout=30, encoding='gbk', errors='replace'
                )
                if result.returncode == 0:
                    QMessageBox.information(self, "Success", "WSL has been shutdown.\nIt will restart automatically on next use.")
                else:
                    QMessageBox.warning(self, "Warning", f"wsl --shutdown returned: {result.stderr}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to restart WSL: {e}")

    def update_config(self):
        """Update config from UI values."""
        self.config.wsl.distro = self.combo_wsl_distro.currentText()
        self.config.wsl.cpu_limit = self.spin_wsl_cpu.value()
        self.config.wsl.memory_limit = self.spin_wsl_memory.value()
        self.config.model.model_path = self.edit_model_path.text()
        self.config.model.venv_path = self.edit_venv_path.text()
        self.config.model.tensor_parallel_size = self.spin_tp.value()
        self.config.model.pipeline_parallel_size = self.spin_pp.value()
        self.config.model.moe_expert_parallel_size = self.spin_moe_ep.value()
        self.config.model.moe_tensor_parallel_size = self.spin_moe_tp.value()
        self.config.model.cuda_visible_devices = self.edit_cuda_devices.text()
        self.config.model.dtype = self.combo_dtype.currentText()


class MemoryConfigTab(QWidget):
    """Memory and KV Cache configuration tab - CRITICAL for memory optimization."""

    def __init__(self, config: AppConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Warning banner
        warning = QLabel(
            "IMPORTANT: free_gpu_memory_fraction=0 means KV cache size is controlled ONLY by max_tokens.\n"
            "This prevents TensorRT-LLM from pre-allocating 90% of free VRAM!"
        )
        warning.setStyleSheet("background-color: #FFF3CD; color: #856404; padding: 10px; font-weight: bold;")
        warning.setWordWrap(True)
        layout.addWidget(warning)

        # KV Cache Memory Fraction
        fraction_group = QGroupBox("GPU Memory Fraction for KV Cache")
        fraction_layout = QVBoxLayout(fraction_group)

        fraction_desc = QLabel(
            "Percentage of FREE GPU memory to allocate for KV cache.\n"
            "0 = Disabled (recommended) - use max_tokens only\n"
            "0.9 = 90% of free VRAM (TensorRT-LLM default - NOT recommended!)"
        )
        fraction_desc.setWordWrap(True)
        fraction_layout.addWidget(fraction_desc)

        fraction_row = QHBoxLayout()
        self.slider_fraction = QSlider(Qt.Orientation.Horizontal)
        self.slider_fraction.setRange(0, 100)
        self.slider_fraction.setValue(int(self.config.kv_cache.free_gpu_memory_fraction * 100))
        self.slider_fraction.valueChanged.connect(self._on_fraction_changed)

        self.spin_fraction = QDoubleSpinBox()
        self.spin_fraction.setRange(0.0, 1.0)
        self.spin_fraction.setSingleStep(0.05)
        self.spin_fraction.setDecimals(2)
        self.spin_fraction.setValue(self.config.kv_cache.free_gpu_memory_fraction)
        self.spin_fraction.valueChanged.connect(self._on_spin_fraction_changed)

        self.lbl_fraction_status = QLabel()
        self._update_fraction_status()

        fraction_row.addWidget(QLabel("0"))
        fraction_row.addWidget(self.slider_fraction, stretch=1)
        fraction_row.addWidget(QLabel("1.0"))
        fraction_row.addWidget(self.spin_fraction)

        fraction_layout.addLayout(fraction_row)
        fraction_layout.addWidget(self.lbl_fraction_status)
        layout.addWidget(fraction_group)

        # Max Tokens
        tokens_group = QGroupBox("KV Cache Max Tokens")
        tokens_layout = QFormLayout(tokens_group)

        self.spin_max_tokens = QSpinBox()
        self.spin_max_tokens.setRange(1024, 10485760)
        self.spin_max_tokens.setSingleStep(1024)
        self.spin_max_tokens.setValue(self.config.kv_cache.max_tokens)
        self.spin_max_tokens.setToolTip("Maximum tokens in KV cache. Recommended: max_seq_len * max_batch_size")

        self.btn_auto_tokens = QPushButton("Auto Calculate")
        self.btn_auto_tokens.clicked.connect(self._auto_calculate_tokens)

        tokens_row = QHBoxLayout()
        tokens_row.addWidget(self.spin_max_tokens)
        tokens_row.addWidget(self.btn_auto_tokens)

        tokens_layout.addRow("Max Tokens:", tokens_row)
        layout.addWidget(tokens_group)

        # Other KV Cache Settings
        other_group = QGroupBox("Other KV Cache Settings")
        other_layout = QFormLayout(other_group)

        self.chk_block_reuse = QCheckBox()
        self.chk_block_reuse.setChecked(self.config.kv_cache.enable_block_reuse)
        self.chk_block_reuse.setToolTip("Enable reuse of KV cache blocks across requests")

        self.combo_kv_dtype = QComboBox()
        self.combo_kv_dtype.addItems(["auto", "fp8", "nvfp4"])
        self.combo_kv_dtype.setCurrentText(self.config.kv_cache.dtype)
        self.combo_kv_dtype.setToolTip("auto=infer from model, fp8=FP8 quantized, nvfp4=FP4 quantized (Blackwell)")

        self.spin_tokens_per_block = QSpinBox()
        self.spin_tokens_per_block.setRange(8, 128)
        self.spin_tokens_per_block.setValue(self.config.kv_cache.tokens_per_block)

        self.spin_host_cache = QSpinBox()
        self.spin_host_cache.setRange(0, 1024 * 1024)  # Up to 1TB in MB
        self.spin_host_cache.setSingleStep(1024)
        self.spin_host_cache.setValue(self.config.kv_cache.host_cache_size // (1024 * 1024))  # Convert to MB
        self.spin_host_cache.setSuffix(" MB")
        self.spin_host_cache.setToolTip("Host RAM for KV cache offloading. 0 = disabled")

        other_layout.addRow("Enable Block Reuse:", self.chk_block_reuse)
        other_layout.addRow("KV Cache Dtype:", self.combo_kv_dtype)
        other_layout.addRow("Tokens Per Block:", self.spin_tokens_per_block)
        other_layout.addRow("Host Cache Size:", self.spin_host_cache)
        layout.addWidget(other_group)

        layout.addStretch()

    def _on_fraction_changed(self, value: int):
        self.spin_fraction.blockSignals(True)
        self.spin_fraction.setValue(value / 100.0)
        self.spin_fraction.blockSignals(False)
        self._update_fraction_status()

    def _on_spin_fraction_changed(self, value: float):
        self.slider_fraction.blockSignals(True)
        self.slider_fraction.setValue(int(value * 100))
        self.slider_fraction.blockSignals(False)
        self._update_fraction_status()

    def _update_fraction_status(self):
        value = self.spin_fraction.value()
        if value == 0:
            status = "DISABLED - KV cache size controlled by max_tokens only (RECOMMENDED)"
            color = "green"
        elif value < 0.5:
            status = f"{int(value * 100)}% of free VRAM will be pre-allocated for KV cache"
            color = "orange"
        else:
            status = f"WARNING: {int(value * 100)}% of free VRAM will be pre-allocated! This may waste memory!"
            color = "red"
        self.lbl_fraction_status.setText(status)
        self.lbl_fraction_status.setStyleSheet(f"color: {color}; font-weight: bold;")

    def _auto_calculate_tokens(self):
        # Get values from inference tab via parent
        parent = self.parent()
        while parent and not isinstance(parent, TRTLLMManager):
            parent = parent.parent()
        if parent:
            parent.inference_tab.update_config()
            max_seq = parent.config_manager.config.inference.max_seq_len
            max_batch = parent.config_manager.config.inference.max_batch_size
            calculated = max_seq * max_batch
            self.spin_max_tokens.setValue(calculated)

    def update_config(self):
        """Update config from UI values."""
        self.config.kv_cache.free_gpu_memory_fraction = self.spin_fraction.value()
        self.config.kv_cache.max_tokens = self.spin_max_tokens.value()
        self.config.kv_cache.enable_block_reuse = self.chk_block_reuse.isChecked()
        self.config.kv_cache.dtype = self.combo_kv_dtype.currentText()
        self.config.kv_cache.tokens_per_block = self.spin_tokens_per_block.value()
        self.config.kv_cache.host_cache_size = self.spin_host_cache.value() * 1024 * 1024  # MB to bytes


class InferenceConfigTab(QWidget):
    """Inference settings tab."""

    def __init__(self, config: AppConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Sequence settings
        seq_group = QGroupBox("Sequence Settings")
        seq_layout = QFormLayout(seq_group)

        self.spin_max_seq = QSpinBox()
        self.spin_max_seq.setRange(128, 262144)
        self.spin_max_seq.setSingleStep(1024)
        self.spin_max_seq.setValue(self.config.inference.max_seq_len)

        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 256)
        self.spin_batch.setValue(self.config.inference.max_batch_size)

        self.lbl_calculated_tokens = QLabel()
        self._update_calculated_tokens()
        self.spin_max_seq.valueChanged.connect(self._update_calculated_tokens)
        self.spin_batch.valueChanged.connect(self._update_calculated_tokens)

        seq_layout.addRow("Max Sequence Length:", self.spin_max_seq)
        seq_layout.addRow("Max Batch Size:", self.spin_batch)
        seq_layout.addRow("Calculated KV Tokens:", self.lbl_calculated_tokens)
        layout.addWidget(seq_group)

        # Max Num Tokens (Prefill limit)
        prefill_group = QGroupBox("Prefill Configuration")
        prefill_group.setToolTip("Controls maximum tokens in a single prefill request")
        prefill_layout = QFormLayout(prefill_group)

        self.spin_max_num_tokens = QSpinBox()
        self.spin_max_num_tokens.setRange(0, 262144)
        self.spin_max_num_tokens.setSingleStep(4096)
        self.spin_max_num_tokens.setValue(self.config.inference.max_num_tokens)
        self.spin_max_num_tokens.setToolTip(
            "Maximum tokens per prefill request (input prompt limit).\n"
            "0 = Auto (uses max_seq_len value).\n"
            "Set this >= your longest expected prompt!\n"
            "Cline/Claude typically sends 10k-30k tokens."
        )

        self.lbl_prefill_hint = QLabel()
        self._update_prefill_hint()
        self.spin_max_num_tokens.valueChanged.connect(self._update_prefill_hint)
        self.spin_max_seq.valueChanged.connect(self._update_prefill_hint)

        prefill_layout.addRow("Max Num Tokens:", self.spin_max_num_tokens)
        prefill_layout.addRow("", self.lbl_prefill_hint)
        layout.addWidget(prefill_group)

        # Features
        feat_group = QGroupBox("Features")
        feat_layout = QFormLayout(feat_group)

        self.chk_chunked = QCheckBox()
        self.chk_chunked.setChecked(self.config.inference.enable_chunked_prefill)
        self.chk_chunked.setToolTip("Split long prompts into chunks for processing")

        self.chk_disable_overlap = QCheckBox()
        self.chk_disable_overlap.setChecked(self.config.inference.disable_overlap_scheduler)
        self.chk_disable_overlap.setToolTip(
            "Disable CPU-GPU overlap scheduler.\n"
            "CHECKED = Lower CPU usage (recommended)\n"
            "UNCHECKED = Max throughput but high CPU usage\n"
            "The overlap scheduler runs CPU work while GPU computes,\n"
            "which increases throughput but causes high CPU utilization."
        )

        self.chk_disable_autotuner = QCheckBox()
        self.chk_disable_autotuner.setChecked(self.config.inference.disable_autotuner)
        self.chk_disable_autotuner.setToolTip(
            "Disable MoE GEMM Autotuner.\n"
            "CHECKED = Skip MoE GEMM profiling (fixes OOM during warmup)\n"
            "UNCHECKED = Enable autotuner for optimal MoE performance\n\n"
            "Enable this if you get 'Can't allocate profile workspace for MoE GEMM profile' error.\n"
            "The autotuner profiles different GEMM tactics during warmup which requires extra VRAM."
        )

        self.chk_cuda_graph = QCheckBox()
        self.chk_cuda_graph.setChecked(self.config.inference.cuda_graph_enabled)
        self.chk_cuda_graph.setToolTip(
            "Enable CUDA Graph optimization.\n"
            "CHECKED = Use CUDA graphs for reduced kernel launch overhead (default)\n"
            "UNCHECKED = Disable CUDA graphs (may help with multi-GPU warmup hangs)\n\n"
            "Try disabling if model loading hangs during CUDA graph warmup phase.\n"
            "CUDA graphs pre-record GPU operations but require extra memory during warmup."
        )

        feat_layout.addRow("Enable Chunked Prefill:", self.chk_chunked)
        feat_layout.addRow("Disable Overlap Scheduler:", self.chk_disable_overlap)
        feat_layout.addRow("Disable MoE Autotuner:", self.chk_disable_autotuner)
        feat_layout.addRow("Enable CUDA Graph:", self.chk_cuda_graph)
        layout.addWidget(feat_group)

        layout.addStretch()

    def _update_calculated_tokens(self):
        calculated = self.spin_max_seq.value() * self.spin_batch.value()
        self.lbl_calculated_tokens.setText(f"{calculated:,} tokens (max_seq_len * max_batch_size)")

    def _update_prefill_hint(self):
        val = self.spin_max_num_tokens.value()
        max_seq = self.spin_max_seq.value()
        if val == 0:
            self.lbl_prefill_hint.setText(f"Auto: will use {max_seq:,} (= max_seq_len)")
            self.lbl_prefill_hint.setStyleSheet("color: #4CAF50;")
        elif val < 32768:
            self.lbl_prefill_hint.setText("Warning: Too small for Cline! Recommend >= 32768")
            self.lbl_prefill_hint.setStyleSheet("color: #FF5722;")
        else:
            self.lbl_prefill_hint.setText(f"OK: Can handle prompts up to {val:,} tokens")
            self.lbl_prefill_hint.setStyleSheet("color: #4CAF50;")

    def update_config(self):
        """Update config from UI values."""
        self.config.inference.max_seq_len = self.spin_max_seq.value()
        self.config.inference.max_batch_size = self.spin_batch.value()
        self.config.inference.max_num_tokens = self.spin_max_num_tokens.value()
        self.config.inference.enable_chunked_prefill = self.chk_chunked.isChecked()
        self.config.inference.disable_overlap_scheduler = self.chk_disable_overlap.isChecked()
        self.config.inference.disable_autotuner = self.chk_disable_autotuner.isChecked()
        self.config.inference.cuda_graph_enabled = self.chk_cuda_graph.isChecked()


class ApiConfigTab(QWidget):
    """API server settings tab with WSL2 port forwarding."""

    def __init__(self, config: AppConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self._wsl_ip: Optional[str] = None
        self._setup_ui()
        self._refresh_status()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # API Server Settings
        api_group = QGroupBox("API Server Settings")
        api_layout = QFormLayout(api_group)

        self.edit_host = QLineEdit(self.config.api.host)
        self.spin_port = QSpinBox()
        self.spin_port.setRange(1, 65535)
        self.spin_port.setValue(self.config.api.port)

        api_layout.addRow("Host:", self.edit_host)
        api_layout.addRow("Port:", self.spin_port)
        layout.addWidget(api_group)

        # WSL2 Port Forwarding
        forward_group = QGroupBox("WSL2 Port Forwarding")
        forward_layout = QVBoxLayout(forward_group)

        # Enable checkbox
        self.chk_port_forward = QCheckBox("Enable automatic port forwarding on service start")
        self.chk_port_forward.setChecked(self.config.api.enable_port_forward)
        self.chk_port_forward.setToolTip(
            "Automatically setup Windows port forwarding when service starts.\n"
            "This exposes WSL2 API port to external network.\n"
            "Uses: netsh interface portproxy add v4tov4"
        )
        forward_layout.addWidget(self.chk_port_forward)

        # Status display
        status_layout = QFormLayout()
        self.lbl_wsl_ip = QLabel("Not detected")
        self.lbl_wsl_ip.setStyleSheet("font-family: monospace;")
        self.lbl_forward_status = QLabel("Unknown")
        self.lbl_forward_status.setStyleSheet("font-family: monospace;")
        status_layout.addRow("WSL2 IP:", self.lbl_wsl_ip)
        status_layout.addRow("Forwarding:", self.lbl_forward_status)
        forward_layout.addLayout(status_layout)

        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_refresh = QPushButton("Refresh Status")
        self.btn_refresh.clicked.connect(self._refresh_status)
        self.btn_setup = QPushButton("Setup Forwarding")
        self.btn_setup.clicked.connect(self._setup_forwarding)
        self.btn_setup.setStyleSheet("background-color: #4CAF50; color: white;")
        self.btn_remove = QPushButton("Remove Forwarding")
        self.btn_remove.clicked.connect(self._remove_forwarding)
        self.btn_remove.setStyleSheet("background-color: #f44336; color: white;")
        btn_layout.addWidget(self.btn_refresh)
        btn_layout.addWidget(self.btn_setup)
        btn_layout.addWidget(self.btn_remove)
        forward_layout.addLayout(btn_layout)

        # Current rules display
        self.txt_rules = QTextEdit()
        self.txt_rules.setReadOnly(True)
        self.txt_rules.setMaximumHeight(100)
        self.txt_rules.setStyleSheet("font-family: monospace; font-size: 10px;")
        self.txt_rules.setPlaceholderText("Current port forwarding rules will be shown here...")
        forward_layout.addWidget(self.txt_rules)

        layout.addWidget(forward_group)
        layout.addStretch()

    def _get_wsl_ip(self) -> Optional[str]:
        """Get WSL2 IP address."""
        try:
            result = subprocess.run(
                ["wsl", "-d", self.config.wsl.distro, "hostname", "-I"],
                capture_output=True, timeout=10, encoding='utf-8', errors='replace'
            )
            if result.returncode == 0 and result.stdout.strip():
                # hostname -I may return multiple IPs, take the first one
                ip = result.stdout.strip().split()[0]
                return ip
        except Exception:
            pass
        return None

    def _get_forwarding_rules(self) -> str:
        """Get current port forwarding rules."""
        try:
            result = subprocess.run(
                ["netsh", "interface", "portproxy", "show", "v4tov4"],
                capture_output=True, timeout=10, encoding='gbk', errors='replace'
            )
            if result.returncode == 0:
                return result.stdout
        except Exception as e:
            return f"Error: {e}"
        return ""

    def _check_forwarding_exists(self, port: int) -> bool:
        """Check if forwarding rule exists for given port."""
        rules = self._get_forwarding_rules()
        return f"0.0.0.0         {port}" in rules or f"*               {port}" in rules

    def _refresh_status(self):
        """Refresh WSL IP and forwarding status."""
        # Get WSL IP
        self._wsl_ip = self._get_wsl_ip()
        if self._wsl_ip:
            self.lbl_wsl_ip.setText(self._wsl_ip)
            self.lbl_wsl_ip.setStyleSheet("font-family: monospace; color: green;")
        else:
            self.lbl_wsl_ip.setText("Not detected (WSL not running?)")
            self.lbl_wsl_ip.setStyleSheet("font-family: monospace; color: red;")

        # Check forwarding status
        port = self.spin_port.value()
        if self._check_forwarding_exists(port):
            self.lbl_forward_status.setText(f"ACTIVE (port {port})")
            self.lbl_forward_status.setStyleSheet("font-family: monospace; color: green; font-weight: bold;")
        else:
            self.lbl_forward_status.setText(f"NOT CONFIGURED (port {port})")
            self.lbl_forward_status.setStyleSheet("font-family: monospace; color: orange;")

        # Show all rules
        rules = self._get_forwarding_rules()
        self.txt_rules.setText(rules if rules.strip() else "No port forwarding rules configured.")

    def _setup_forwarding(self):
        """Setup port forwarding from Windows to WSL2."""
        if not self._wsl_ip:
            self._wsl_ip = self._get_wsl_ip()
        if not self._wsl_ip:
            QMessageBox.warning(self, "Error", "Cannot detect WSL2 IP address.\nMake sure WSL is running.")
            return

        port = self.spin_port.value()

        # Remove existing rule first (if any)
        subprocess.run(
            ["netsh", "interface", "portproxy", "delete", "v4tov4",
             f"listenport={port}", "listenaddress=0.0.0.0"],
            capture_output=True, timeout=10
        )

        # Add new rule
        try:
            result = subprocess.run(
                ["netsh", "interface", "portproxy", "add", "v4tov4",
                 f"listenport={port}", "listenaddress=0.0.0.0",
                 f"connectport={port}", f"connectaddress={self._wsl_ip}"],
                capture_output=True, timeout=10, encoding='gbk', errors='replace'
            )
            if result.returncode == 0:
                QMessageBox.information(
                    self, "Success",
                    f"Port forwarding configured:\n"
                    f"0.0.0.0:{port} -> {self._wsl_ip}:{port}\n\n"
                    f"External clients can now access the API."
                )
            else:
                QMessageBox.warning(self, "Error", f"Failed to setup forwarding:\n{result.stderr}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Exception: {e}")

        self._refresh_status()

    def _remove_forwarding(self):
        """Remove port forwarding rule."""
        port = self.spin_port.value()
        try:
            result = subprocess.run(
                ["netsh", "interface", "portproxy", "delete", "v4tov4",
                 f"listenport={port}", "listenaddress=0.0.0.0"],
                capture_output=True, timeout=10, encoding='gbk', errors='replace'
            )
            if result.returncode == 0:
                QMessageBox.information(self, "Success", f"Port forwarding removed for port {port}.")
            else:
                QMessageBox.warning(self, "Warning", f"No rule found or error:\n{result.stderr}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Exception: {e}")

        self._refresh_status()

    def setup_forwarding_silent(self) -> bool:
        """Setup port forwarding without dialogs (for auto-setup on service start)."""
        if not self._wsl_ip:
            self._wsl_ip = self._get_wsl_ip()
        if not self._wsl_ip:
            return False

        port = self.spin_port.value()

        # Remove existing rule first
        subprocess.run(
            ["netsh", "interface", "portproxy", "delete", "v4tov4",
             f"listenport={port}", "listenaddress=0.0.0.0"],
            capture_output=True, timeout=10
        )

        # Add new rule
        try:
            result = subprocess.run(
                ["netsh", "interface", "portproxy", "add", "v4tov4",
                 f"listenport={port}", "listenaddress=0.0.0.0",
                 f"connectport={port}", f"connectaddress={self._wsl_ip}"],
                capture_output=True, timeout=10
            )
            return result.returncode == 0
        except Exception:
            return False

    def remove_forwarding_silent(self) -> bool:
        """Remove port forwarding without dialogs (for auto-cleanup on service stop)."""
        port = self.spin_port.value()
        try:
            result = subprocess.run(
                ["netsh", "interface", "portproxy", "delete", "v4tov4",
                 f"listenport={port}", "listenaddress=0.0.0.0"],
                capture_output=True, timeout=10
            )
            return result.returncode == 0
        except Exception:
            return False

    def update_config(self):
        """Update config from UI values."""
        self.config.api.host = self.edit_host.text()
        self.config.api.port = self.spin_port.value()
        self.config.api.enable_port_forward = self.chk_port_forward.isChecked()


class GPUMonitorTab(QWidget):
    """GPU monitoring tab."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        self.gpu_table = QTableWidget()
        self.gpu_table.setColumnCount(6)
        self.gpu_table.setHorizontalHeaderLabels([
            "Index", "Name", "Used (MB)", "Total (MB)", "Util (%)", "Temp (C)"
        ])
        self.gpu_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.gpu_table.setAlternatingRowColors(True)

        layout.addWidget(self.gpu_table)

    def update_gpu_info(self, gpus: List[Dict]):
        """Update GPU table with new data."""
        self.gpu_table.setRowCount(len(gpus))
        for i, gpu in enumerate(gpus):
            self.gpu_table.setItem(i, 0, QTableWidgetItem(gpu['index']))
            self.gpu_table.setItem(i, 1, QTableWidgetItem(gpu['name']))
            self.gpu_table.setItem(i, 2, QTableWidgetItem(gpu['memory_used']))
            self.gpu_table.setItem(i, 3, QTableWidgetItem(gpu['memory_total']))
            self.gpu_table.setItem(i, 4, QTableWidgetItem(gpu['utilization']))
            self.gpu_table.setItem(i, 5, QTableWidgetItem(gpu['temperature']))


# =============================================================================
# Main Window
# =============================================================================

class TRTLLMManager(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.config_manager = ConfigManager()
        self.process_manager = ProcessManager()
        self.gpu_thread: Optional[GPUMonitorThread] = None
        self._setup_ui()
        self._connect_signals()
        self._start_gpu_monitor()

    def _setup_ui(self):
        self.setWindowTitle("TensorRT-LLM Service Manager - Memory Optimized")
        self.setMinimumSize(1200, 850)
        self.setFont(QFont("Microsoft YaHei", 10))

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Create tabs
        tabs = QTabWidget()

        self.service_tab = ServiceControlTab()
        self.model_tab = ModelConfigTab(self.config_manager.config)
        self.memory_tab = MemoryConfigTab(self.config_manager.config)
        self.inference_tab = InferenceConfigTab(self.config_manager.config)
        self.api_tab = ApiConfigTab(self.config_manager.config)
        self.gpu_tab = GPUMonitorTab()

        tabs.addTab(self.service_tab, "Service Control")
        tabs.addTab(self.model_tab, "Model Config")
        tabs.addTab(self.memory_tab, "Memory & KV Cache")
        tabs.addTab(self.inference_tab, "Inference")
        tabs.addTab(self.api_tab, "API Settings")
        tabs.addTab(self.gpu_tab, "GPU Monitor")

        layout.addWidget(tabs)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self._update_status("Ready")

    def _connect_signals(self):
        self.service_tab.btn_start.clicked.connect(self._start_service)
        self.service_tab.btn_stop.clicked.connect(self._stop_service)
        self.service_tab.btn_restart.clicked.connect(self._restart_service)
        self.service_tab.btn_force_kill.clicked.connect(self._force_kill)
        self.service_tab.btn_save.clicked.connect(self._save_config)

    def _start_gpu_monitor(self):
        self.gpu_thread = GPUMonitorThread(self.config_manager.config.wsl.distro)
        self.gpu_thread.gpu_updated.connect(self.gpu_tab.update_gpu_info)
        self.gpu_thread.start()

    def _update_status(self, msg: str):
        running = self.process_manager.is_running
        status = "Running" if running else "Stopped"
        self.status_bar.showMessage(f"{msg} | Service: {status}")

    def _update_config_from_ui(self):
        """Collect all configuration from UI tabs."""
        self.model_tab.update_config()
        self.memory_tab.update_config()
        self.inference_tab.update_config()
        self.api_tab.update_config()

    def _log(self, msg: str):
        self.service_tab.append_log(msg)

    def _start_service(self):
        if self.process_manager.is_running:
            self._log("Service is already running")
            return

        self._update_config_from_ui()
        config = self.config_manager.config

        # Validate
        if not config.model.model_path:
            QMessageBox.warning(self, "Error", "Please set model path!")
            return
        if not config.model.venv_path:
            QMessageBox.warning(self, "Error", "Please set venv path!")
            return

        self._log("Starting TensorRT-LLM service...")
        self._log(f"Model: {config.model.model_path}")
        self._log(f"TP: {config.model.tensor_parallel_size}, PP: {config.model.pipeline_parallel_size}")
        self._log(f"KV Cache: free_gpu_memory_fraction={config.kv_cache.free_gpu_memory_fraction}, max_tokens={config.kv_cache.max_tokens}")

        if self.process_manager.start(config, self._log):
            self.service_tab.set_running(
                True,
                str(self.process_manager.process.pid),
                f"http://{config.api.host}:{config.api.port}"
            )
            self._update_status("Service started")

            # Auto-setup port forwarding if enabled
            if config.api.enable_port_forward:
                self._log("Setting up WSL2 port forwarding...")
                if self.api_tab.setup_forwarding_silent():
                    wsl_ip = self.api_tab._wsl_ip
                    self._log(f"Port forwarding: 0.0.0.0:{config.api.port} -> {wsl_ip}:{config.api.port}")
                else:
                    self._log("WARNING: Failed to setup port forwarding. External access may not work.")
        else:
            self._log("Failed to start service")

    def _stop_service(self):
        if not self.process_manager.is_running:
            self._log("No running service")
            return

        self._log("Stopping service...")
        self.process_manager.stop(self.config_manager.config.wsl.distro)
        self.service_tab.set_running(False)
        self._update_status("Service stopped")
        self._log("Service stopped")

    def _restart_service(self):
        self._log("Restarting service...")
        self._stop_service()
        self._start_service()

    def _force_kill(self):
        self._log("Force killing all related processes...")
        self.process_manager.force_kill(
            self.config_manager.config.wsl.distro,
            self.config_manager.config.api.port
        )
        self.service_tab.set_running(False)
        self._update_status("Force killed")
        self._log("All related processes killed")

    def _terminate_wsl(self):
        reply = QMessageBox.question(
            self, "Terminate WSL",
            f"This will completely terminate WSL distro '{self.config_manager.config.wsl.distro}'.\n"
            "All processes in this distro will be killed!\n\n"
            "This is the nuclear option - use only if Force Kill doesn't work.\n\n"
            "Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._log(f"Terminating WSL distro: {self.config_manager.config.wsl.distro}")
            success = self.process_manager.terminate_wsl(self.config_manager.config.wsl.distro)
            self.service_tab.set_running(False)
            if success:
                self._update_status("WSL terminated")
                self._log("WSL distro terminated successfully")
            else:
                self._update_status("WSL terminate failed")
                self._log("Failed to terminate WSL distro")

    def _save_config(self):
        self._update_config_from_ui()
        self.config_manager.save()
        self._log("Configuration saved")
        self._update_status("Configuration saved")

    def closeEvent(self, event):
        if self.process_manager.is_running:
            reply = QMessageBox.question(
                self, "Confirm Exit",
                "Service is still running. Stop service and exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._stop_service()
            else:
                event.ignore()
                return

        # Auto-save configuration before closing
        self._update_config_from_ui()
        self.config_manager.save()

        if self.gpu_thread:
            self.gpu_thread.stop()
            self.gpu_thread.wait()

        event.accept()


# =============================================================================
# Application Entry Point
# =============================================================================

def setup_dark_theme(app: QApplication):
    """Apply dark theme to application."""
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Text, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(0, 0, 0))
    app.setPalette(palette)


def main():
    app = QApplication(sys.argv)
    setup_dark_theme(app)

    window = TRTLLMManager()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
