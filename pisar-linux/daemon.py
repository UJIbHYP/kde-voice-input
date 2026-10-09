#!/usr/bin/env python3
"""Giga Pisar для Linux — голосовой ввод с Alt.

Зажми Alt → говори → отпусти → текст появится в буфере и будет вставлен
в активное поле (виртуальная клавиатура через X11).
Трей показывает состояние. Модель хранится в ~/.giga/model.
"""
import os
import sys
import subprocess
import threading
import tempfile
import time

import numpy as np
import sounddevice as sd
from pynput import keyboard as pk

# Подтягиваем giga_core и прочие модули строго из соседней папки server
# (при запуске из репы) либо из pisar-linux/ (после установки).
# ВАЖНО: когда демон запускается как ~/.local/bin/giga-pisar,
# каталог скрипта сам становистся sys.path[0] и затеняет server/
# старыми копиями. Поэтому служебные модули грузим из server
# ЯВНО через importlib, а не обычным import.
# Определяем реальный путь к директории скрипта, разрешая симлинки.
# Это позволяет без проблем запускать скрипт через симлинк из ~/.local/bin/
_HERE = os.path.dirname(os.path.realpath(__file__))
_SERVER = None
for _p in [
    os.path.join(_HERE, "..", "server"),   # при запуске из репы
    _HERE,                                # если запущен из pisar-linux/
]:
    if os.path.isdir(_p) and os.path.isfile(os.path.join(_p, "giga_core.py")):
        _SERVER = os.path.abspath(_p)
        sys.path.insert(0, _SERVER)
        break
else:
    sys.exit("❌ не найден giga_core.py — проверьте установку")

import importlib.util


def _load_module(name: str):
    """Загрузить модуль из серверной папки или pisar-linux (защита от теневых копий)."""
    # Сначала пробуем server/ - там giga_core
    if name == "giga_core":
        spec = importlib.util.spec_from_file_location(
            name, os.path.join(_SERVER, f"{name}.py")
        )
        if spec is None:
            raise ImportError(f"не найден {name}.py в {_SERVER}")
    else:
        # tray.py, commands.py, clipboard.py ищем в pisar-linux
        spec = importlib.util.spec_from_file_location(
            name, os.path.join(_HERE, f"{name}.py")
        )
        if spec is None:
            raise ImportError(f"не найден {name}.py в {_HERE}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


from giga_core import Engine, SAMPLE_RATE, MAX_CHUNK, find_model_dir

try:
    set_clipboard = _load_module("clipboard").set_clipboard
    Tray = _load_module("tray").Tray
    CommandProcessor = _load_module("commands").CommandProcessor
except (ImportError, FileNotFoundError):
    import subprocess

    def set_clipboard(text: str):
        try:
            subprocess.run(["wl-copy"], input=text.encode(), check=True)
        except FileNotFoundError:
            subprocess.run(["xclip", "-selection", "clipboard"],
                           input=text.encode(), check=True)

    class Tray:
        def set_state(self, state: str): pass

    class CommandProcessor:
        def process(self, text: str) -> bool: return False
        def reload(self): pass


# ──────────────────────────── настройки ────────────────────────────

HOTKEY = "alt"                 # какая клавиша запускает запись
DURATION = 1.0                 # секунд ждать перед стартом после нажатия
RECORD_SEC = 30                # максимум секунд записи
VOL_THRESH = 0.01              # порог голоса — ниже считаем тишиной


class PisarDaemon:
    def __init__(self):
        self.tray = Tray(on_quit=self.request_quit)
        self.cmdproc = CommandProcessor(on_command=self._on_command_triggered)
        self.tray.set_commands(self.cmdproc.count)
        self.tray.set_commands(self.cmdproc.count)
        self.engine = Engine(model_dir=find_model_dir(""))
        self.recording = False
        self.buffer: list[np.ndarray] = []
        self.lock = threading.Lock()
        self._transcribe_lock = threading.Lock()
        self.stream = None
        self._last_activity = 0.0
        self.uinput_dev = None
        try:
            import subprocess
            subprocess.run(["kwtype", "--version"], check=True, capture_output=True)
            self._kwtype_available = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            self._kwtype_available = False

        try:
            import evdev.uinput as ui
            # KEY_LEFTCTRL = 29, KEY_V = 47
            self.uinput_dev = ui.UInput({ui.ecodes.EV_KEY: [29, 47]}, name='giga-pisar-virtkbd')
            print("[pisar] ✓ виртуальная клавиатура uinput инициализирована", file=sys.stderr)
        except Exception as e:
            print(f"[pisar] ⚠ не удалось инициализировать uinput (вставка может не работать): {e}", file=sys.stderr)

    def _record_callback(self, indata, frames, timeinfo, status):
        chunk = indata[:, 0].astype(np.float32) / 32768.0
        with self.lock:
            if not self.recording:
                return
            self.buffer.append(chunk)
            # Ограничиваем буфер — не даём расти больше RECORD_SEC
            max_chunks = int(RECORD_SEC * SAMPLE_RATE / 4096) + 1
            if len(self.buffer) > max_chunks:
                self.buffer = self.buffer[-max_chunks:]
            elif len(self.buffer) % 500 == 1:
                print(f"[pisar] звук: {len(self.buffer)} чанков", file=sys.stderr)
        if status:
            print(f"[pisar] audio status: {status}", file=sys.stderr)

    def _start_recording(self):
        with self.lock:
            if self.recording:
                return
            self.buffer.clear()
            self.recording = True
            self._last_activity = time.time()
        self.tray.set_state("recording")
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16",
            callback=self._record_callback, blocksize=4096,
        )
        self.stream.start()

    def _stop_recording(self):
        with self.lock:
            if not self.recording:
                return None
            self.recording = False
            chunks = list(self.buffer)
            self.buffer.clear()
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        if not chunks:
            print("[pisar] ⚠ записанного звука нет", file=sys.stderr)
            return None

        wav = np.concatenate(chunks)
        # отбросить хвост тишины
        vol = np.abs(wav)
        above = np.where(vol > VOL_THRESH)[0]
        if len(above) == 0:
            return None
        last_speech = int(above[-1] + 0.3 * SAMPLE_RATE)
        wav = wav[:last_speech]

        if len(wav) < SAMPLE_RATE * 0.5:
            return None  # слишком коротко

        self.tray.set_state("transcribing")
        threading.Thread(target=self._transcribe, args=(wav,), daemon=True).start()
        return "transcribing"

    def _paste_text(self, text: str):
        """Автоматически вставляет текст в активное поле через kwtype."""
        if not self._kwtype_available:
            # kwtype нет — пробуем фоллбэк uinput
            import evdev.uinput as ui
            if self.uinput_dev is None:
                return
            time.sleep(0.15)
            try:
                self.uinput_dev.write(ui.ecodes.EV_KEY, 29, 1)  # Ctrl
                self.uinput_dev.write(ui.ecodes.EV_KEY, 47, 1)  # V
                self.uinput_dev.write(ui.ecodes.EV_KEY, 47, 0)
                self.uinput_dev.write(ui.ecodes.EV_KEY, 29, 0)
                self.uinput_dev.syn()
            except Exception as e:
                print(f"[pisar] ⚠ вставка не удалась: {e}", file=sys.stderr)
            return

        try:
            time.sleep(0.15)
            subprocess.run(["kwtype", text], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"[pisar] ⚠ kwtype упал: {e}", file=sys.stderr)

    def _transcribe(self, wav: np.ndarray):
        path = None
        try:
            with self._transcribe_lock:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    path = f.name
                import wave as wavemod
                with wavemod.open(path, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(SAMPLE_RATE)
                    wf.writeframes((wav * 32767).astype(np.int16).tobytes())

                text = self.engine.transcribe(path)
        except Exception as e:
            print(f"[pisar] ❌ {e}", file=sys.stderr)
        finally:
            if path:
                os.unlink(path)
            self.tray.set_state("idle")

        if text:
            print(f"[pisar] ✓ {text!r}", file=sys.stderr)
            # Сначала проверяем голосовые команды: если это команда,
            # не трогаем буфер обмена и не вставляем текст.
            triggered = self.cmdproc.process(text)
            if triggered:
                print(f"[pisar] ⚡ команда распознана: {text!r}", file=sys.stderr)
                # Для команд не копируем текст в буфер и не вставляем автоматически
            else:
                if self.tray.save_clipboard:
                    set_clipboard(text)
                    print("[pisar]   текст сохранён в буфер обмена", file=sys.stderr)
                if self.tray.auto_paste:
                    threading.Thread(
                        target=self._paste_text, args=(text,), daemon=True
                    ).start()

    def _on_command_triggered(self, text: str, cmd):
        """Сработала голосовая команда — показываем уведомление."""
        self.tray.show_command_notification(text)

    # ──────────────────────────── жизненный цикл ────────────────────────

    def request_quit(self):
        print("[pisar] выход через трей", file=sys.stderr)
        self._trigger_quit()

    def _trigger_quit(self):
        if not hasattr(self, '_quit_done'):
            return
        if self._quit_done.is_set():
            return
        self._quit_done.set()
        if self._quit_done.is_set():
            return
        self._quit_done.set()

        try:
            with self.lock:
                self.recording = False
            if self.stream:
                try:
                    self.stream.stop()
                    self.stream.close()
                except Exception:
                    pass
                self.stream = None
        except Exception as e:
            print(f"[pisar] ⚠ остановка записи при выходе: {e}", file=sys.stderr)

        # Гасим рестартованный цикл Gtk.main() — иначе поток может
        # снова войти в Gtk.main() и блокироваться на выходе
        self._stop_gtk.set()

        try:
            self.tray.stop()    # корректно гасит индикатор в GTK-потоке
        except Exception:
            pass

    def run(self):
        print("[pisar] старт. Зажми Alt для записи.", file=sys.stderr)
        self._quit_done = threading.Event()
        self._stop_gtk = threading.Event()

        # GTK цикл в отдельном потоке — для трей-иконки
        from gi.repository import Gtk

        def _gtk_loop():
            while not self._stop_gtk.is_set():
                try:
                    Gtk.main()
                except Exception as e:
                    print(f"[pisar] ⚠ GTK поток упал: {e}, рестарт через 5с", file=sys.stderr)
                    time.sleep(5)
            print("[pisar] GTK цикл остановлен", file=sys.stderr)

        self.tray.set_state("idle")
        gtk_thread = threading.Thread(target=_gtk_loop, daemon=True)
        gtk_thread.start()

        # Слушаем Alt через pynput (работает без root)
        alt_pressed = False

        def on_press(key):
            nonlocal alt_pressed
            print(f"[pisar] press: {key!r}", file=sys.stderr)
            if key == pk.Key.alt:
                alt_pressed = True
                self._start_recording()

        def on_release(key):
            nonlocal alt_pressed
            print(f"[pisar] release: {key!r}", file=sys.stderr)
            if key == pk.Key.alt:
                alt_pressed = False
                self._stop_recording()

        listener = pk.Listener(on_press=on_press, on_release=on_release)
        listener.start()

        try:
            sd.default.latency = "low"
            while not self._quit_done.is_set():
                time.sleep(0.2)
        except KeyboardInterrupt:
            print("\n[pisar] выход", file=sys.stderr)
        finally:
            self._trigger_quit()
            self._stop_gtk.set()
            listener.stop()
            print("[pisar] работа завершена", file=sys.stderr)


if __name__ == "__main__":
    PisarDaemon().run()
