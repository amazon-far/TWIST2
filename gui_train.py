#!/usr/bin/env python3
"""TWIST2 G1 23-DoF — Bash launcher GUI (Tkinter, stdlib only).

Runs each training / deployment shell script as a subprocess and tails its
stdout into a scrolling log panel. No visualisation; just buttons + fields.

Run:  python3 gui_train.py    (or)    bash gui_train.sh
"""

from __future__ import annotations

import os
import queue
import shlex
import signal
import subprocess
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

ROOT = Path(__file__).resolve().parent

# (label, track, default_cmd_template) — {exptid}, {ckpt}, {teacher} placeholders
COMMANDS = [
    # IsaacGym track
    ("IG · Teacher 학습",   "isaacgym", f"bash {ROOT}/train.sh {{exptid}} cuda:0"),
    ("IG · Student 학습",   "isaacgym", f"bash {ROOT}/train.sh {{exptid}} cuda:0"),
    ("IG · ONNX 변환",      "isaacgym", f"bash {ROOT}/to_onnx.sh {{ckpt}}"),
    # Isaac Lab track
    ("Lab · Teacher 학습",  "isaaclab", f"bash {ROOT}/isaaclab_train/scripts/train_teacher.sh {{exptid}}"),
    ("Lab · Student 학습",  "isaaclab", f"bash {ROOT}/isaaclab_train/scripts/train_student.sh {{exptid}} {{teacher}}"),
    ("Lab · Play(시각화)",  "isaaclab", f"bash {ROOT}/isaaclab_train/scripts/play.sh {{ckpt}}"),
    # Deployment (shared)
    ("Sim2Sim",             "deploy",   f"bash {ROOT}/sim2sim.sh"),
    ("Sim2Real",            "deploy",   f"bash {ROOT}/sim2real.sh"),
    ("Motion Server",       "deploy",   f"bash {ROOT}/run_motion_server.sh"),
    ("Teleop (PICO→Redis)", "deploy",   f"bash {ROOT}/teleop.sh"),
]


class Launcher(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("TWIST2 G1 23-DoF Launcher")
        self.geometry("980x640")

        self._procs: list[subprocess.Popen] = []
        self._log_q: "queue.Queue[str]" = queue.Queue()

        self._build_inputs()
        self._build_buttons()
        self._build_log()

        self.after(50, self._drain_log)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------ UI

    def _build_inputs(self):
        frm = ttk.LabelFrame(self, text="공통 파라미터")
        frm.pack(fill="x", padx=8, pady=6)

        self.exptid = tk.StringVar(value="exp001")
        self.ckpt = tk.StringVar(value="")
        self.teacher = tk.StringVar(value="")

        for i, (lbl, var, browse) in enumerate([
            ("exptid", self.exptid, False),
            ("ckpt path", self.ckpt, True),
            ("teacher ckpt", self.teacher, True),
        ]):
            ttk.Label(frm, text=lbl, width=14).grid(row=i, column=0, sticky="w", padx=6, pady=3)
            ttk.Entry(frm, textvariable=var, width=80).grid(row=i, column=1, sticky="we", padx=4)
            if browse:
                ttk.Button(frm, text="…", width=3,
                           command=lambda v=var: self._browse(v)).grid(row=i, column=2, padx=4)
        frm.columnconfigure(1, weight=1)

    def _build_buttons(self):
        nb = ttk.Notebook(self)
        nb.pack(fill="x", padx=8, pady=4)

        groups: dict[str, ttk.Frame] = {}
        for track, label in [("isaacgym", "IsaacGym"),
                             ("isaaclab", "Isaac Lab"),
                             ("deploy",   "Deploy / Teleop")]:
            f = ttk.Frame(nb)
            nb.add(f, text=label)
            groups[track] = f

        for label, track, tmpl in COMMANDS:
            f = groups[track]
            ttk.Button(f, text=label, width=28,
                       command=lambda t=tmpl, l=label: self._run(t, l)
                       ).pack(side="left", padx=4, pady=8)

        ctrl = ttk.Frame(self)
        ctrl.pack(fill="x", padx=8, pady=2)
        ttk.Button(ctrl, text="모든 작업 중단",
                   command=self._kill_all).pack(side="right")
        ttk.Button(ctrl, text="로그 지우기",
                   command=lambda: self.log.delete("1.0", "end")).pack(side="right", padx=6)

    def _build_log(self):
        self.log = scrolledtext.ScrolledText(self, font=("Monospace", 10),
                                             bg="#1e1e1e", fg="#d4d4d4",
                                             insertbackground="white")
        self.log.pack(fill="both", expand=True, padx=8, pady=6)

    # ------------------------------------------------------------------ actions

    def _browse(self, var: tk.StringVar):
        path = filedialog.askopenfilename(
            initialdir=str(ROOT / "assets" / "ckpts") if (ROOT / "assets" / "ckpts").exists() else str(ROOT),
            filetypes=[("Checkpoints", "*.pt *.onnx"), ("All", "*.*")],
        )
        if path:
            var.set(path)

    def _run(self, tmpl: str, label: str):
        try:
            cmd = tmpl.format(exptid=self.exptid.get().strip(),
                              ckpt=self.ckpt.get().strip(),
                              teacher=self.teacher.get().strip())
        except KeyError as e:
            messagebox.showerror("입력 부족", f"필수 필드가 비었습니다: {e}")
            return
        if "{" in cmd:
            messagebox.showerror("입력 부족", f"필수 필드가 비었습니다.\n{cmd}")
            return

        self._log(f"\n$ {cmd}\n")
        try:
            p = subprocess.Popen(
                shlex.split(cmd),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                bufsize=1, text=True, cwd=str(ROOT),
                preexec_fn=os.setsid,
            )
        except FileNotFoundError as e:
            self._log(f"[error] {e}\n")
            return
        self._procs.append(p)
        threading.Thread(target=self._reader, args=(p, label), daemon=True).start()

    def _reader(self, p: subprocess.Popen, label: str):
        assert p.stdout
        for line in iter(p.stdout.readline, ""):
            self._log_q.put(line)
        p.wait()
        self._log_q.put(f"[{label}] exit code = {p.returncode}\n")

    def _drain_log(self):
        try:
            while True:
                line = self._log_q.get_nowait()
                self._log(line)
        except queue.Empty:
            pass
        self.after(80, self._drain_log)

    def _log(self, s: str):
        self.log.insert("end", s)
        self.log.see("end")

    def _kill_all(self):
        for p in self._procs:
            if p.poll() is None:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass
        self._log("[!] sent SIGTERM to all running jobs\n")

    def _on_close(self):
        if any(p.poll() is None for p in self._procs):
            if not messagebox.askyesno("종료", "실행 중인 작업이 있습니다. 종료할까요?"):
                return
            self._kill_all()
        self.destroy()


if __name__ == "__main__":
    Launcher().mainloop()
