import os
import json
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, font, filedialog
from datetime import datetime

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None


class AIClient:

    PROVIDERS = {
        "Groq AI": {
            "base_url": "https://api.groq.com/openai/v1",
            "default_model": "llama-3.3-70b-versatile",
            "models": [
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "mixtral-8x7b-32768",
                "gemma2-9b-it",
                "llama3-70b-8192",
                "llama3-8b-8192",
                "llama-guard-3-8b",
            ],
        },
        "OpenAI": {
            "base_url": "https://api.openai.com/v1",
            "default_model": "gpt-4o",
            "models": [
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4-turbo",
                "gpt-3.5-turbo",
            ],
        },
        "Custom": {
            "base_url": "",
            "default_model": "",
            "models": [],
        },
    }

    def __init__(self):
        self.client = None
        self.current_provider = "Groq AI"
        self.api_key = ""
        self.base_url = self.PROVIDERS["Groq AI"]["base_url"]
        self.model = self.PROVIDERS["Groq AI"]["default_model"]
        self.temperature = 0.7
        self.max_tokens = 4096
        self.system_prompt = "You are a helpful, knowledgeable AI assistant. Answer any question the user asks thoroughly and accurately."

    def configure(self, provider, api_key, base_url=None, model=None):
        self.current_provider = provider
        self.api_key = api_key

        if base_url:
            self.base_url = base_url
        else:
            self.base_url = self.PROVIDERS.get(provider, {}).get("base_url", "")

        if model:
            self.model = model
        elif provider in self.PROVIDERS:
            self.model = self.PROVIDERS[provider].get("default_model", "")
        else:
            self.model = ""

        if not self.api_key or not self.base_url or not self.model:
            return False, "API Key, Base URL, and Model are required."

        try:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
            return True, "Connected to {} - Model: {}".format(provider, self.model)
        except Exception as e:
            self.client = None
            return False, "Configuration error: {}".format(str(e))

    def send_message(self, messages, on_token=None, on_done=None, on_error=None):
        if not self.client:
            if on_error:
                on_error("Client not configured. Please check your settings.")
            return

        try:
            api_messages = [{"role": "system", "content": self.system_prompt}]
            for m in messages:
                api_messages.append({"role": m["role"], "content": m["content"]})

            kwargs = {
                "model": self.model,
                "messages": api_messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": True,
            }

            response = self.client.chat.completions.create(**kwargs)
            full_content = ""

            for chunk in response:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        content = delta.content
                        full_content += content
                        if on_token:
                            on_token(content)

            if on_done:
                on_done(full_content)

        except Exception as e:
            if on_error:
                error_msg = str(e)
                if "401" in error_msg or "unauthorized" in error_msg.lower():
                    error_msg = "Authentication failed. Check your API key."
                elif "404" in error_msg:
                    error_msg = "Endpoint not found. Check your Base URL and model name."
                elif "429" in error_msg or "rate" in error_msg.lower():
                    error_msg = "Rate limited. Please wait before sending another request."
                on_error(error_msg)


class AIAssistantUI:

    def __init__(self, root):
        self.root = root
        self.root.title("AI Assistant Pro - Groq - OpenAI - Custom")
        self.root.geometry("1100x720")
        self.root.minsize(900, 600)

        self.setup_theme()

        self.client = AIClient()
        self.messages = []
        self.current_response = ""
        self.is_streaming = False

        self.build_menu()
        self.build_main_layout()
        self.build_settings_panel()
        self.build_chat_area()
        self.build_status_bar()

        self.load_config()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def setup_theme(self):
        self.colors = {
            "bg_dark": "#0707f5",
            "bg_medium": "#16213e",
            "bg_light": "#0f3460",
            "accent": "#ef0e34",
            "accent_hover": "#ff6b81",
            "text_primary": "#eaeaea",
            "text_secondary": "#a0a0b0",
            "input_bg": "#2a2a4a",
            "user_bubble": "#0f3460",
            "ai_bubble": "#1a1a3e",
            "success": "#2ecc71",
            "warning": "#f39c12",
            "error": "#e74c3c",
        }

        style = ttk.Style()
        style.theme_use("clam")

        style.configure("TButton",
                        background=self.colors["bg_light"],
                        foreground=self.colors["text_primary"],
                        borderwidth=0,
                        focusthickness=0,
                        focuscolor="none",
                        padding=(10, 5))
        style.map("TButton",
                  background=[("active", self.colors["accent"]),
                              ("pressed", self.colors["accent_hover"])])

        style.configure("Accent.TButton",
                        background=self.colors["accent"],
                        foreground="#ffffff",
                        borderwidth=0,
                        padding=(15, 6))
        style.map("Accent.TButton",
                  background=[("active", self.colors["accent_hover"]),
                              ("pressed", "#c0392b")])

        style.configure("TLabel",
                        background=self.colors["bg_dark"],
                        foreground=self.colors["text_primary"])
        style.configure("Header.TLabel",
                        font=("Segoe UI", 14, "bold"),
                        foreground=self.colors["accent"])
        style.configure("Title.TLabel",
                        font=("Segoe UI", 16, "bold"),
                        foreground=self.colors["text_primary"])
        style.configure("TEntry",
                        fieldbackground=self.colors["input_bg"],
                        foreground=self.colors["text_primary"],
                        borderwidth=0,
                        padding=5)
        style.configure("TCombobox",
                        fieldbackground=self.colors["input_bg"],
                        foreground=self.colors["text_primary"],
                        arrowcolor=self.colors["text_primary"])
        style.map("TCombobox",
                  fieldbackground=[("readonly", self.colors["input_bg"])])

        self.root.configure(bg=self.colors["bg_dark"])

    def build_menu(self):
        menubar = tk.Menu(self.root, bg=self.colors["bg_medium"],
                          fg=self.colors["text_primary"],
                          activebackground=self.colors["accent"],
                          activeforeground="#ffffff",
                          font=("Segoe UI", 10))

        file_menu = tk.Menu(menubar, tearoff=0, bg=self.colors["bg_medium"],
                            fg=self.colors["text_primary"],
                            activebackground=self.colors["accent"],
                            activeforeground="#ffffff")
        file_menu.add_command(label="Export Chat (.txt)", command=self.export_chat_txt)
        file_menu.add_command(label="Export Chat (.json)", command=self.export_chat_json)
        file_menu.add_separator()
        file_menu.add_command(label="Clear Conversation", command=self.clear_conversation)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_close)
        menubar.add_cascade(label="File", menu=file_menu)

        settings_menu = tk.Menu(menubar, tearoff=0, bg=self.colors["bg_medium"],
                                fg=self.colors["text_primary"],
                                activebackground=self.colors["accent"],
                                activeforeground="#ffffff")
        settings_menu.add_command(label="Toggle Settings Panel", command=self.toggle_settings_panel)
        settings_menu.add_separator()
        settings_menu.add_command(label="Save Configuration", command=self.save_config)
        settings_menu.add_command(label="Load Configuration", command=self.load_config_from_file)
        menubar.add_cascade(label="Settings", menu=settings_menu)

        help_menu = tk.Menu(menubar, tearoff=0, bg=self.colors["bg_medium"],
                            fg=self.colors["text_primary"],
                            activebackground=self.colors["accent"],
                            activeforeground="#ffffff")
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.root.config(menu=menubar)

    def build_main_layout(self):
        self.main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_pane.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.settings_frame = tk.Frame(self.main_pane, bg=self.colors["bg_medium"],
                                       width=320)
        self.main_pane.add(self.settings_frame, weight=0)

        self.chat_frame = tk.Frame(self.main_pane, bg=self.colors["bg_dark"])
        self.main_pane.add(self.chat_frame, weight=1)

    def build_settings_panel(self):
        frame = self.settings_frame

        canvas = tk.Canvas(frame, bg=self.colors["bg_medium"],
                           highlightthickness=0, bd=0)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        self.settings_inner = tk.Frame(canvas, bg=self.colors["bg_medium"])

        self.settings_inner.bind("<Configure>",
                                 lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.settings_inner, anchor="nw", width=300)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        sf = self.settings_inner

        tk.Label(sf, text="PROVIDER", font=("Segoe UI", 13, "bold"),
                 bg=self.colors["bg_medium"], fg=self.colors["accent"]).pack(anchor="w", pady=(15, 5), padx=15)

        self.provider_var = tk.StringVar(value="Groq AI")
        self.provider_combo = ttk.Combobox(sf, textvariable=self.provider_var,
                                           values=list(AIClient.PROVIDERS.keys()),
                                           state="readonly", width=35)
        self.provider_combo.pack(padx=15, pady=(0, 10), fill="x")
        self.provider_combo.bind("<<ComboboxSelected>>", self.on_provider_change)

        tk.Label(sf, text="API KEY", font=("Segoe UI", 11, "bold"),
                 bg=self.colors["bg_medium"], fg=self.colors["text_primary"]).pack(anchor="w", padx=15, pady=(10, 2))

        api_key_frame = tk.Frame(sf, bg=self.colors["bg_medium"])
        api_key_frame.pack(padx=15, fill="x")

        self.api_key_var = tk.StringVar()
        self.api_key_entry = tk.Entry(api_key_frame, textvariable=self.api_key_var,
                                      bg=self.colors["input_bg"], fg=self.colors["text_primary"],
                                      insertbackground=self.colors["text_primary"],
                                      relief="flat", font=("Consolas", 10),
                                      show="*", bd=0, highlightthickness=0)
        self.api_key_entry.pack(side="left", fill="x", expand=True, ipady=6)

        self.toggle_key_btn = tk.Button(api_key_frame, text="SHOW", font=("Segoe UI", 9),
                                        bg=self.colors["bg_light"], fg=self.colors["text_primary"],
                                        relief="flat", bd=0, cursor="hand2",
                                        activebackground=self.colors["accent"],
                                        command=self.toggle_api_key_visibility)
        self.toggle_key_btn.pack(side="right", padx=(5, 0))

        tk.Label(sf, text="BASE URL", font=("Segoe UI", 11, "bold"),
                 bg=self.colors["bg_medium"], fg=self.colors["text_primary"]).pack(anchor="w", padx=15, pady=(10, 2))

        self.base_url_var = tk.StringVar(value=AIClient.PROVIDERS["Groq AI"]["base_url"])
        self.base_url_entry = tk.Entry(sf, textvariable=self.base_url_var,
                                       bg=self.colors["input_bg"], fg=self.colors["text_primary"],
                                       insertbackground=self.colors["text_primary"],
                                       relief="flat", font=("Consolas", 10),
                                       bd=0, highlightthickness=0)
        self.base_url_entry.pack(padx=15, fill="x", ipady=6)

        tk.Label(sf, text="MODEL", font=("Segoe UI", 11, "bold"),
                 bg=self.colors["bg_medium"], fg=self.colors["text_primary"]).pack(anchor="w", padx=15, pady=(10, 2))

        self.model_var = tk.StringVar(value=AIClient.PROVIDERS["Groq AI"]["default_model"])
        self.model_combo = ttk.Combobox(sf, textvariable=self.model_var,
                                        values=AIClient.PROVIDERS["Groq AI"]["models"],
                                        width=35)
        self.model_combo.pack(padx=15, pady=(0, 10), fill="x")

        tk.Label(sf, text="TEMPERATURE", font=("Segoe UI", 11, "bold"),
                 bg=self.colors["bg_medium"], fg=self.colors["text_primary"]).pack(anchor="w", padx=15, pady=(10, 2))

        temp_frame = tk.Frame(sf, bg=self.colors["bg_medium"])
        temp_frame.pack(padx=15, fill="x")

        self.temp_var = tk.DoubleVar(value=0.7)
        self.temp_scale = tk.Scale(temp_frame, from_=0.0, to=2.0, resolution=0.05,
                                    orient=tk.HORIZONTAL, variable=self.temp_var,
                                    bg=self.colors["bg_medium"],
                                    fg=self.colors["text_primary"],
                                    troughcolor=self.colors["input_bg"],
                                    activebackground=self.colors["accent"],
                                    highlightthickness=0, bd=0,
                                    length=180, sliderlength=20)
        self.temp_scale.pack(side="left")

        self.temp_label = tk.Label(temp_frame, text="0.70",
                                   bg=self.colors["bg_medium"],
                                   fg=self.colors["text_secondary"],
                                   font=("Consolas", 10))
        self.temp_label.pack(side="left", padx=(10, 0))
        self.temp_var.trace_add("write", lambda *_: self.temp_label.config(text="{:.2f}".format(self.temp_var.get())))

        tk.Label(sf, text="MAX TOKENS", font=("Segoe UI", 11, "bold"),
                 bg=self.colors["bg_medium"], fg=self.colors["text_primary"]).pack(anchor="w", padx=15, pady=(10, 2))

        self.max_tokens_var = tk.StringVar(value="4096")
        tk.Entry(sf, textvariable=self.max_tokens_var,
                 bg=self.colors["input_bg"], fg=self.colors["text_primary"],
                 insertbackground=self.colors["text_primary"],
                 relief="flat", font=("Consolas", 10),
                 bd=0, highlightthickness=0).pack(padx=15, fill="x", ipady=6)

        tk.Label(sf, text="SYSTEM PROMPT", font=("Segoe UI", 11, "bold"),
                 bg=self.colors["bg_medium"], fg=self.colors["text_primary"]).pack(anchor="w", padx=15, pady=(10, 2))

        self.system_prompt_text = tk.Text(sf, height=4, width=30,
                                           bg=self.colors["input_bg"],
                                           fg=self.colors["text_primary"],
                                           insertbackground=self.colors["text_primary"],
                                           relief="flat", bd=0,
                                           font=("Segoe UI", 10),
                                           highlightthickness=0)
        self.system_prompt_text.pack(padx=15, fill="x", pady=(0, 10))
        self.system_prompt_text.insert("1.0", self.client.system_prompt)

        self.connect_btn = ttk.Button(sf, text="CONNECT", style="Accent.TButton",
                                      command=self.connect_to_api)
        self.connect_btn.pack(pady=15, padx=15, fill="x")

        self.connection_status = tk.Label(sf, text="NOT CONNECTED",
                                          bg=self.colors["bg_medium"],
                                          fg=self.colors["warning"],
                                          font=("Segoe UI", 9))
        self.connection_status.pack(pady=(0, 15))

        tk.Label(sf, text="ACTIONS", font=("Segoe UI", 11, "bold"),
                 bg=self.colors["bg_medium"], fg=self.colors["accent"]).pack(anchor="w", padx=15, pady=(5, 5))

        action_frame = tk.Frame(sf, bg=self.colors["bg_medium"])
        action_frame.pack(padx=15, pady=(0, 15), fill="x")

        tk.Button(action_frame, text="CLEAR CHAT",
                  bg=self.colors["bg_light"], fg=self.colors["text_primary"],
                  relief="flat", bd=0, cursor="hand2",
                  activebackground=self.colors["accent"],
                  command=self.clear_conversation,
                  font=("Segoe UI", 9)).pack(side="left", fill="x", expand=True, padx=(0, 3), ipady=4)

        tk.Button(action_frame, text="EXPORT",
                  bg=self.colors["bg_light"], fg=self.colors["text_primary"],
                  relief="flat", bd=0, cursor="hand2",
                  activebackground=self.colors["accent"],
                  command=self.export_chat_txt,
                  font=("Segoe UI", 9)).pack(side="left", fill="x", expand=True, padx=(3, 0), ipady=4)

    def build_chat_area(self):
        frame = self.chat_frame

        header = tk.Frame(frame, bg=self.colors["bg_dark"])
        header.pack(fill="x", pady=(0, 5))

        tk.Label(header, text="AI ASSISTANT PRO",
                 font=("Segoe UI", 16, "bold"),
                 bg=self.colors["bg_dark"], fg=self.colors["accent"]).pack(side="left", padx=10)

        self.provider_indicator = tk.Label(header, text="[Groq AI]",
                                           font=("Segoe UI", 10),
                                           bg=self.colors["bg_dark"],
                                           fg=self.colors["text_secondary"])
        self.provider_indicator.pack(side="left", padx=(5, 0))

        chat_display_frame = tk.Frame(frame, bg=self.colors["bg_dark"])
        chat_display_frame.pack(fill="both", expand=True, padx=5, pady=5)

        self.chat_display = tk.Text(chat_display_frame,
                                     wrap=tk.WORD,
                                     bg=self.colors["bg_dark"],
                                     fg=self.colors["text_primary"],
                                     insertbackground=self.colors["text_primary"],
                                     relief="flat", bd=0,
                                     font=("Segoe UI", 11),
                                     highlightthickness=0,
                                     state=tk.DISABLED,
                                     padx=15, pady=10)
        self.chat_display.pack(side="left", fill="both", expand=True)

        chat_scroll = ttk.Scrollbar(chat_display_frame, orient="vertical",
                                    command=self.chat_display.yview)
        chat_scroll.pack(side="right", fill="y")
        self.chat_display.configure(yscrollcommand=chat_scroll.set)

        self.chat_display.tag_config("user_tag",
                                     foreground="#4fc3f7",
                                     font=("Segoe UI", 10, "bold"),
                                     spacing3=5)
        self.chat_display.tag_config("ai_tag",
                                     foreground=self.colors["accent"],
                                     font=("Segoe UI", 10, "bold"),
                                     spacing3=5)
        self.chat_display.tag_config("system_tag",
                                     foreground=self.colors["text_secondary"],
                                     font=("Segoe UI", 9, "italic"),
                                     spacing3=3)
        self.chat_display.tag_config("user_msg",
                                     foreground=self.colors["text_primary"],
                                     font=("Segoe UI", 11),
                                     spacing1=2, spacing3=10,
                                     lmargin1=20)
        self.chat_display.tag_config("ai_msg",
                                     foreground="#e0e0e0",
                                     font=("Segoe UI", 11),
                                     spacing1=2, spacing3=10,
                                     lmargin1=20)
        self.chat_display.tag_config("code_block",
                                     background=self.colors["input_bg"],
                                     foreground="#f8f8f2",
                                     font=("Consolas", 10),
                                     spacing1=4, spacing3=4,
                                     lmargin1=25, lmargin2=25)

        input_frame = tk.Frame(frame, bg=self.colors["bg_dark"])
        input_frame.pack(fill="x", padx=5, pady=(0, 10))

        text_bg = tk.Frame(input_frame, bg=self.colors["input_bg"],
                           highlightthickness=0, bd=0)
        text_bg.pack(fill="x", pady=(0, 8))

        self.input_text = tk.Text(text_bg, height=3, wrap=tk.WORD,
                                   bg=self.colors["input_bg"],
                                   fg=self.colors["text_primary"],
                                   insertbackground=self.colors["text_primary"],
                                   relief="flat", bd=0,
                                   font=("Segoe UI", 11),
                                   highlightthickness=0,
                                   padx=10, pady=8)
        self.input_text.pack(fill="x")
        self.input_text.bind("<Return>", self.on_enter_key)
        self.input_text.bind("<Shift-Return>", lambda e: None)
        self.input_text.focus_set()

        btn_frame = tk.Frame(input_frame, bg=self.colors["bg_dark"])
        btn_frame.pack(fill="x")

        self.send_btn = ttk.Button(btn_frame, text="SEND (ENTER)", style="Accent.TButton",
                                   command=self.send_message)
        self.send_btn.pack(side="right")

        self.stop_btn = ttk.Button(btn_frame, text="STOP", style="TButton",
                                   command=self.stop_streaming)
        self.stop_btn.pack(side="right", padx=(0, 10))
        self.stop_btn.pack_forget()

        tk.Button(btn_frame, text="NEW LINE (SHIFT+ENTER)",
                  bg=self.colors["bg_dark"], fg=self.colors["text_secondary"],
                  relief="flat", bd=0, cursor="hand2",
                  activebackground=self.colors["bg_light"],
                  font=("Segoe UI", 8)).pack(side="left")

    def build_status_bar(self):
        self.status_bar = tk.Frame(self.root, bg=self.colors["bg_medium"], height=25)
        self.status_bar.pack(fill="x", side="bottom")

        self.status_label = tk.Label(self.status_bar, text="Ready",
                                      bg=self.colors["bg_medium"],
                                      fg=self.colors["text_secondary"],
                                      font=("Segoe UI", 9))
        self.status_label.pack(side="left", padx=10)

        self.token_count_label = tk.Label(self.status_bar, text="Tokens: 0",
                                           bg=self.colors["bg_medium"],
                                           fg=self.colors["text_secondary"],
                                           font=("Segoe UI", 9))
        self.token_count_label.pack(side="right", padx=10)

    def on_provider_change(self, event=None):
        provider = self.provider_var.get()
        info = AIClient.PROVIDERS.get(provider, {})

        if provider == "Custom":
            self.base_url_var.set("")
            self.model_combo["values"] = []
            self.model_var.set("")
        else:
            self.base_url_var.set(info.get("base_url", ""))
            models = info.get("models", [])
            self.model_combo["values"] = models
            if models:
                self.model_var.set(info.get("default_model", models[0]))

    def on_enter_key(self, event):
        if not event.state & 0x0001:
            self.send_message()
            return "break"
        return None

    def toggle_api_key_visibility(self):
        if self.api_key_entry.cget("show") == "*":
            self.api_key_entry.config(show="")
            self.toggle_key_btn.config(text="HIDE")
        else:
            self.api_key_entry.config(show="*")
            self.toggle_key_btn.config(text="SHOW")

    def toggle_settings_panel(self):
        if self.settings_frame.winfo_viewable():
            self.main_pane.forget(self.settings_frame)
        else:
            self.main_pane.insert(0, self.settings_frame, weight=0)

    def connect_to_api(self):
        provider = self.provider_var.get()
        api_key = self.api_key_var.get().strip()
        base_url = self.base_url_var.get().strip()
        model = self.model_var.get().strip()

        system_prompt = self.system_prompt_text.get("1.0", "end-1c").strip()
        if system_prompt:
            self.client.system_prompt = system_prompt

        success, msg = self.client.configure(provider, api_key, base_url, model)
        if success:
            self.client.temperature = self.temp_var.get()
            try:
                self.client.max_tokens = int(self.max_tokens_var.get())
            except ValueError:
                self.client.max_tokens = 4096

            self.connection_status.config(text="CONNECTED - {}".format(msg), fg=self.colors["success"])
            self.provider_indicator.config(text="[{} - {}]".format(provider, model),
                                           fg=self.colors["success"])
            self.status_label.config(text="Connected - {}".format(msg), fg=self.colors["success"])
            self.append_system_message("Connected to {} | Model: {}".format(provider, model))
        else:
            self.connection_status.config(text="FAILED - {}".format(msg), fg=self.colors["error"])
            self.status_label.config(text="Connection failed", fg=self.colors["error"])

    def send_message(self):
        content = self.input_text.get("1.0", "end-1c").strip()
        if not content:
            return

        if not self.client.client:
            self.append_system_message("Not connected. Configure API settings and click Connect first.")
            return

        if self.is_streaming:
            return

        self.input_text.delete("1.0", "end")

        self.append_user_message(content)

        self.messages.append({"role": "user", "content": content})

        self.is_streaming = True
        self.send_btn.config(text="SENDING...")
        self.stop_btn.pack(side="right", padx=(0, 10))

        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, "\nAI ASSISTANT\n", "ai_tag")
        self.current_response = ""

        self.ai_msg_start = self.chat_display.index("end-1c")
        self.chat_display.insert(tk.END, "\n", "ai_tag")
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)

        threading.Thread(target=self.stream_response, daemon=True).start()

    def stream_response(self):
        def on_token(token):
            self.root.after(0, self._append_stream_token, token)

        def on_done(full_content):
            self.root.after(0, self._on_stream_done, full_content)

        def on_error(error_msg):
            self.root.after(0, self._on_stream_error, error_msg)

        self.client.send_message(self.messages, on_token=on_token, on_done=on_done, on_error=on_error)

    def _append_stream_token(self, token):
        self.current_response += token
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, token, "ai_msg")
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)

    def _on_stream_done(self, full_content):
        self.messages.append({"role": "assistant", "content": full_content})
        self.is_streaming = False
        self.send_btn.config(text="SEND (ENTER)")
        self.stop_btn.pack_forget()
        self.status_label.config(text="Response received ({} chars)".format(len(full_content)),
                                 fg=self.colors["success"])
        self.token_count_label.config(text="Tokens: {}".format(len(full_content) // 4))

    def _on_stream_error(self, error_msg):
        self.is_streaming = False
        self.send_btn.config(text="SEND (ENTER)")
        self.stop_btn.pack_forget()

        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, "\nError: {}\n".format(error_msg), "system_tag")
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)

        self.connection_status.config(text="ERROR - {}".format(error_msg), fg=self.colors["error"])
        self.status_label.config(text="Error: {}".format(error_msg), fg=self.colors["error"])

    def stop_streaming(self):
        self.is_streaming = False
        self._on_stream_error("Streaming stopped by user")

    def append_user_message(self, content):
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, "\nYOU\n", "user_tag")
        self.chat_display.insert(tk.END, "{}\n".format(content), "user_msg")
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)

    def append_system_message(self, content):
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, "\n{}\n".format(content), "system_tag")
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)

    def clear_conversation(self):
        if self.messages and not messagebox.askyesno("Clear Chat", "Clear the entire conversation?"):
            return

        self.messages.clear()
        self.current_response = ""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete("1.0", tk.END)
        self.chat_display.config(state=tk.DISABLED)
        self.append_system_message("Conversation cleared. Ready for new questions.")
        self.status_label.config(text="Conversation cleared")

    def export_chat_txt(self):
        if not self.messages:
            messagebox.showinfo("Export", "No messages to export.")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                                  filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if not file_path:
            return

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("AI Assistant Pro - Export\n")
                f.write("{}\n\n".format("=" * 60))
                for msg in self.messages:
                    if msg["role"] == "user":
                        role = "YOU"
                    else:
                        role = "AI ASSISTANT"
                    f.write("{}:\n{}\n\n".format(role, msg["content"]))
            self.status_label.config(text="Exported to {}".format(os.path.basename(file_path)))
            messagebox.showinfo("Export", "Chat exported successfully to:\n{}".format(file_path))
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    def export_chat_json(self):
        if not self.messages:
            messagebox.showinfo("Export", "No messages to export.")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".json",
                                                  filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if not file_path:
            return

        try:
            data = {
                "provider": self.client.current_provider,
                "model": self.client.model,
                "exported_at": datetime.now().isoformat(),
                "messages": self.messages
            }
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self.status_label.config(text="Exported to {}".format(os.path.basename(file_path)))
            messagebox.showinfo("Export", "Chat exported successfully to:\n{}".format(file_path))
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    def save_config(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".json",
                                                  filetypes=[("JSON files", "*.json")],
                                                  initialfile="ai_assistant_config.json")
        if not file_path:
            return

        config = {
            "provider": self.provider_var.get(),
            "api_key": self.api_key_var.get(),
            "base_url": self.base_url_var.get(),
            "model": self.model_var.get(),
            "temperature": self.temp_var.get(),
            "max_tokens": self.max_tokens_var.get(),
            "system_prompt": self.system_prompt_text.get("1.0", "end-1c"),
        }
        try:
            with open(file_path, "w") as f:
                json.dump(config, f, indent=2)
            self.status_label.config(text="Configuration saved")
            messagebox.showinfo("Saved", "Configuration saved successfully.")
        except Exception as e:
            messagebox.showerror("Error", "Failed to save: {}".format(e))

    def load_config_from_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if not file_path:
            return
        self.load_config(file_path)

    def load_config(self, file_path=None):
        if file_path is None:
            default_path = "ai_assistant_config.json"
            if os.path.exists(default_path):
                file_path = default_path
            else:
                return

        try:
            with open(file_path, "r") as f:
                config = json.load(f)

            self.provider_var.set(config.get("provider", "Groq AI"))
            self.api_key_var.set(config.get("api_key", ""))
            self.base_url_var.set(config.get("base_url", ""))
            self.model_var.set(config.get("model", ""))
            self.temp_var.set(config.get("temperature", 0.7))
            self.max_tokens_var.set(config.get("max_tokens", "4096"))

            system_prompt = config.get("system_prompt", "")
            if system_prompt:
                self.system_prompt_text.delete("1.0", "end")
                self.system_prompt_text.insert("1.0", system_prompt)

            self.on_provider_change()
            self.status_label.config(text="Config loaded from {}".format(os.path.basename(file_path)))
        except Exception as e:
            messagebox.showerror("Load Error", "Failed to load config: {}".format(e))

    def show_about(self):
        about_text = """AI Assistant Pro

Version 1.0
Built with Python, tkinter, OpenAI SDK

Supports:
- Groq AI (llama, mixtral, gemma)
- OpenAI (GPT-4, GPT-3.5)
- Any OpenAI-compatible API

No content restrictions.
Full streaming support.
Dark modern UI.

Author: HackerAI
"""
        messagebox.showinfo("About AI Assistant Pro", about_text)

    def on_close(self):
        if messagebox.askokcancel("Exit", "Are you sure you want to exit?"):
            self.root.destroy()


if __name__ == "__main__":
    if not OPENAI_AVAILABLE:
        print("=" * 60)
        print("Required package 'openai' is not installed.")
        print("Install it with:  pip install openai")
        print("=" * 60)
        response = input("Would you like to install it now? (y/n): ")
        if response.lower() == "y":
            import subprocess
            subprocess.check_call(["pip", "install", "openai"])
            print("\nopenai installed. Restarting application...\n")
            from openai import OpenAI
        else:
            print("Please install it manually: pip install openai")
            exit(1)

    root = tk.Tk()
    app = AIAssistantUI(root)
    root.mainloop()
