"""
mcp_server/server.py — Model Context Protocol Server for ESP32 Microscope

Exposes 5 tools over JSON-RPC 2.0 / stdio:
  capture_frame         — fetch JPEG from ESP32, return base64
  stream_status         — fetch /status JSON from ESP32
  analyze_current_frame — capture + Claude Vision analysis
  adjust_camera         — POST brightness/contrast to ESP32
  save_and_label        — save captured frame + metadata to dataset/

Setup (Claude Desktop claude_desktop_config.json):
  {
    "mcpServers": {
      "microscope": {
        "command": "python",
        "args": ["/path/to/mcp_server/server.py"]
      }
    }
  }
"""

import base64
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [MCP] %(levelname)s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# Load config
CONFIG_PATH = Path(__file__).parent / "config.json"
with open(CONFIG_PATH) as f:
    CONFIG = json.load(f)

ESP32_BASE  = "http://" + CONFIG["esp32_ip"] + ":" + str(CONFIG["esp32_port"])
DATASET_DIR = Path(CONFIG["dataset_output_path"])
DATASET_DIR.mkdir(parents=True, exist_ok=True)
TIMEOUT = CONFIG.get("capture_timeout_s", 5)


# ESP32 helpers
def esp32_get(path: str) -> requests.Response:
    return requests.get(ESP32_BASE + path, timeout=TIMEOUT)


def esp32_post(path: str, payload: dict) -> requests.Response:
    return requests.post(ESP32_BASE + path, json=payload, timeout=TIMEOUT)


# Tool implementations
def tool_capture_frame() -> dict[str, Any]:
    """Capture a JPEG from the ESP32 /capture endpoint and return base64-encoded bytes."""
    try:
        resp = esp32_get("/capture")
        resp.raise_for_status()
        b64 = base64.standard_b64encode(resp.content).decode()
        return {"success": True, "image_base64": b64, "size_bytes": len(resp.content)}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def tool_stream_status() -> dict[str, Any]:
    """Fetch system status JSON from the ESP32 /status endpoint."""
    try:
        resp = esp32_get("/status")
        resp.raise_for_status()
        return {"success": True, **resp.json()}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def tool_analyze_current_frame() -> dict[str, Any]:
    """Capture a frame and analyze it using Claude Vision API."""
    try:
        import anthropic

        capture = tool_capture_frame()
        if not capture["success"]:
            return {"success": False, "error": "Capture failed: " + capture.get("error", "")}

        client = anthropic.Anthropic(api_key=CONFIG["anthropic_api_key"])
        message = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1200,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": capture["image_base64"],
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "You are a biomedical microscopy expert. "
                                "Analyze this microscope slide image and describe: "
                                "1) Visible biological structures, "
                                "2) Morphological features and their significance, "
                                "3) Any anomalies or points of interest, "
                                "4) Suggested follow-up observations."
                            ),
                        },
                    ],
                }
            ],
        )
        return {
            "success": True,
            "analysis": message.content[0].text,
            "image_base64": capture["image_base64"],
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def tool_adjust_camera(brightness: int = 0, contrast: int = 0) -> dict[str, Any]:
    """
    POST brightness and contrast settings to the ESP32.
    Both values must be in the range -4 to +4.
    """
    try:
        resp = esp32_post("/settings", {"brightness": brightness, "contrast": contrast})
        resp.raise_for_status()
        return {"success": True, "brightness": brightness, "contrast": contrast}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def tool_save_and_label(label: str, notes: str = "") -> dict[str, Any]:
    """
    Capture the current frame and save it with label metadata to the dataset folder.
    label: class name for the specimen, e.g. red_blood_cell or bacteria.
    notes: optional free-text observation notes.
    """
    try:
        capture = tool_capture_frame()
        if not capture["success"]:
            return {"success": False, "error": "Capture failed"}

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_name  = label + "_" + timestamp + ".jpg"
        meta_name = label + "_" + timestamp + ".json"

        img_path = DATASET_DIR / img_name
        img_path.write_bytes(base64.standard_b64decode(capture["image_base64"]))

        meta = {
            "label":     label,
            "notes":     notes,
            "timestamp": timestamp,
            "image":     img_name,
        }
        (DATASET_DIR / meta_name).write_text(json.dumps(meta, indent=2))

        return {"success": True, "saved_image": img_name, "metadata": meta}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


# Tool registry
TOOLS = {
    "capture_frame":         (tool_capture_frame,        []),
    "stream_status":         (tool_stream_status,         []),
    "analyze_current_frame": (tool_analyze_current_frame, []),
    "adjust_camera":         (tool_adjust_camera,         ["brightness", "contrast"]),
    "save_and_label":        (tool_save_and_label,        ["label", "notes"]),
}

TOOL_SCHEMAS = {
    "capture_frame": {
        "description": "Capture a single JPEG frame from the ESP32 microscope camera.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    "stream_status": {
        "description": "Get current system status including IP, FPS, and free heap from the ESP32.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    "analyze_current_frame": {
        "description": "Capture a frame and run Claude Vision biomedical analysis on it.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    "adjust_camera": {
        "description": "Adjust ESP32 camera brightness and contrast. Range is -4 to +4 for both parameters.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "brightness": {
                    "type": "integer",
                    "minimum": -4,
                    "maximum": 4,
                    "default": 0,
                    "description": "Brightness level from -4 darkest to +4 brightest.",
                },
                "contrast": {
                    "type": "integer",
                    "minimum": -4,
                    "maximum": 4,
                    "default": 0,
                    "description": "Contrast level from -4 lowest to +4 highest.",
                },
            },
        },
    },
    "save_and_label": {
        "description": "Capture and save a labeled image to the dataset folder.",
        "inputSchema": {
            "type": "object",
            "required": ["label"],
            "properties": {
                "label": {
                    "type": "string",
                    "description": "Class label for the specimen, for example bacteria or red_blood_cell.",
                },
                "notes": {
                    "type": "string",
                    "description": "Optional free-text observation notes.",
                },
            },
        },
    },
}


# JSON-RPC 2.0 dispatcher
def handle_request(req: dict) -> dict:
    req_id  = req.get("id")
    method  = req.get("method", "")
    params  = req.get("params", {})

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "microscope-mcp", "version": "1.0.0"},
            },
        }

    if method == "tools/list":
        tools_list = [{"name": name, **schema} for name, schema in TOOL_SCHEMAS.items()]
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": tools_list}}

    if method == "tools/call":
        tool_name = params.get("name")
        tool_args = params.get("arguments", {})

        if tool_name not in TOOLS:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": "Unknown tool: " + str(tool_name)},
            }

        fn, _ = TOOLS[tool_name]
        try:
            result = fn(**tool_args)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(result, indent=2)}]
                },
            }
        except Exception as exc:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32603, "message": str(exc)},
            }

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": "Method not found: " + str(method)},
    }


def main() -> None:
    logger.info("MCP Microscope Server starting. ESP32 target: %s", ESP32_BASE)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req  = json.loads(line)
            resp = handle_request(req)
            print(json.dumps(resp), flush=True)
        except json.JSONDecodeError as exc:
            err = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error: " + str(exc)},
            }
            print(json.dumps(err), flush=True)
        except Exception as exc:
            err = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32603, "message": "Internal error: " + str(exc)},
            }
            print(json.dumps(err), flush=True)


if __name__ == "__main__":
    main()