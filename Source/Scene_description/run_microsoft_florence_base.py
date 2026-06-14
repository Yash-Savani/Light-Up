import cv2
import torch
import time
import os
import warnings
import psutil
from PIL import Image

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import transformers
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
from transformers.models.roberta.tokenization_roberta import RobertaTokenizer

def patch_tokenizer():
    old_init = RobertaTokenizer.__init__
    def new_init(self, *args, **kwargs):
        old_init(self, *args, **kwargs)
        if not hasattr(self, "additional_special_tokens"):
            self.additional_special_tokens = []
        if not hasattr(self, "image_token"):
            self.image_token = "<image>"
    RobertaTokenizer.__init__ = new_init
patch_tokenizer()

transformers.configuration_utils.PretrainedConfig.forced_bos_token_id = None
transformers.PreTrainedModel._supports_sdpa = False
transformers.PreTrainedModel._supports_flash_attn_2 = False

from transformers import AutoProcessor, AutoModelForCausalLM


def smart_truncate(text, max_chars=50):
    """Truncate at word boundary, never mid-word."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > 0:
        return truncated[:last_space]
    return truncated


class FlorenceTester:
    def __init__(self):
        self.model_id = "microsoft/Florence-2-base"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32

        print(f"\n--- LOADING MODEL: {self.model_id.upper()} ---")

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                dtype=self.dtype,
                attn_implementation="eager"
            ).to(self.device)

            if hasattr(self.model, "language_model"):
                lm = self.model.language_model
                shared_embed = lm.model.shared if hasattr(lm.model, "shared") else lm.model.encoder.embed_tokens
                lm.model.encoder.embed_tokens.weight = shared_embed.weight
                lm.model.decoder.embed_tokens.weight = shared_embed.weight
                lm.lm_head.weight = shared_embed.weight
                lm.tie_weights()

            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=True
            )

            self.model.eval()
            print(f"SUCCESS: Florence-2 initialized and weights tied.")
        except Exception as e:
            print(f"ERROR DURING LOADING: {e}")
            raise e

    def get_system_usage(self):
        process = psutil.Process(os.getpid())
        ram_mb = process.memory_info().rss / (1024 * 1024)
        cpu_p = psutil.cpu_percent(interval=None)
        return ram_mb, cpu_p

    def get_description(self, frame):
        start_time = time.time()
        ram, cpu = self.get_system_usage()

        try:
            frame_resized = cv2.resize(frame, (768, 768))
            image = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))

            prompt = "<CAPTION>"
            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=32,
                    num_beams=1,        # Greedy decoding — fastest
                    do_sample=False,
                    use_cache=False
                )

            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

            parsed_answer = self.processor.post_process_generation(
                generated_text,
                task=prompt,
                image_size=(image.width, image.height)
            )

            raw_caption = parsed_answer.get(prompt, parsed_answer.get("<CAPTION>", str(parsed_answer)))
            caption = smart_truncate(raw_caption.strip(), max_chars=50)
            dur = time.time() - start_time

            return caption, dur, ram, cpu

        except Exception as e:
            dur = time.time() - start_time
            import traceback
            traceback.print_exc()
            return f"Inference Error: {str(e)}", dur, ram, cpu


def main():
    try:
        tester = FlorenceTester()
    except Exception:
        return

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    res_label = "1280x720"

    log_file = open("performance_log.txt", "a")
    log_file.write(f"\n{'='*60}\n")
    log_file.write(f"SESSION START: {time.ctime()}\n")
    log_file.write(f"MODEL: microsoft/Florence-2-base\n")
    log_file.write(f"RESOLUTION: {res_label} | HARDWARE: {'cuda' if torch.cuda.is_available() else 'cpu'}\n")
    log_file.write(f"{'-'*60}\n")

    prev_time = 0
    print(f"\nSystem ready. Press 'd' to Describe, 'q' to Quit.")

    while True:
        curr_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        cv2.putText(frame, f"FPS: {int(fps)} | FLORENCE-2", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Res: {res_label} HD", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('Member 3 - Florence-2 Benchmark', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('d'):
            print("Analyzing...")
            desc, dur, ram, cpu = tester.get_description(frame)

            timestamp = time.strftime('%H:%M:%S')
            log_entry = (f"[{timestamp}] {desc}\n"
                         f" > Latency: {dur:.2f}s | RAM: {ram:.1f}MB | CPU: {cpu}% | Device: {'cuda' if torch.cuda.is_available() else 'cpu'}\n")

            print(log_entry)
            log_file.write(log_entry)
            log_file.flush()

        elif key == ord('q'):
            break

    log_file.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
