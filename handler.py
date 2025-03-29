from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import PeftModel
import torch
from PIL import Image
import io
import base64
import os
import logging
from maestro.trainer.models.qwen_2_5_vl.inference import predict
from transformers import BitsAndBytesConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('endpoint_handler.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

class EndpointHandler:
    """Custom handler for processing image inputs and extracting overlay text."""
    
    def __init__(self, path=""):
        """Initialize the model and processor for inference."""
        try:
            token = os.getenv("HF_TOKEN")
            if not token:
                raise ValueError("HF_TOKEN environment variable is not set.")

            from huggingface_hub import login
            login(token=token)
            
            self.processor = AutoProcessor.from_pretrained("MohammedSameerSyed/FinetunedQWEN")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            base_model = AutoModelForVision2Seq.from_pretrained(
                "Qwen/Qwen2.5-VL-3B-Instruct", quantization_config=quantization_config
            )
            self.model = PeftModel.from_pretrained(base_model, "MohammedSameerSyed/FinetunedQWEN")
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()
            
            for param in self.model.parameters():
                param.requires_grad = False
            
            self.SYSTEM_MESSAGE = (
                "You are analyzing an image that may contain overlay text added by video or graphic editing. "
                "Your task is to extract only the overlay text that is deliberately overlaid on the image, "
                "avoiding any background noise such as interface text, headers, or form inputs. "
                "Focus on text that appears visually distinct and is designed to grab attention, such as captions, "
                "titles, or promotional messages. "
                "Do not extract text embedded naturally in the background, on interfaces, or forms. "
                "Preserve the text exactly as it appears, including capitalization, misspellings, and unusual characters. "
                "If no such overlay text exists in the image, return {none} "
                "Focus on identifying Overlay text and ensuring the output adheres to the requested JSON structure. "
                "Provide only the JSON output based on the extracted information."
            )
            
            self.DEFAULT_PREFIX = "You are an assistant that extracts Overlay text from an image."
        
        except Exception as init_error:
            logger.critical(f"Initialization failed: {init_error}")
            raise

    def preprocess_image(self, image_data: str) -> Image.Image:
        """Convert base64-encoded image to PIL Image."""
        try:
            if not image_data or len(image_data) < 100:
                logger.warning("Invalid or empty image data")
                raise ValueError("Image data is too short")

            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            if image.width < 10 or image.height < 10:
                logger.warning("Image too small for processing")
                raise ValueError("Image dimensions are too small")
            
            return image
        
        except Exception as e:
            logger.error(f"Image preprocessing error: {e}")
            raise

    def __call__(self, inputs: dict) -> dict:
        """Process inputs and return extracted overlay text."""
        try:
            if not inputs or 'inputs' not in inputs:
                return {"error": "Invalid input", "received_keys": list(inputs.keys())}
            
            prefix = inputs.get("prefix", self.DEFAULT_PREFIX)
            image_data = inputs.get("inputs")

            image = self.preprocess_image(image_data)

            with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
                torch.cuda.empty_cache()
                generated_suffix = predict(
                    model=self.model,
                    processor=self.processor,
                    image=image,
                    prefix=prefix,
                    system_message=self.SYSTEM_MESSAGE,
                    device=self.device
                )

                if not generated_suffix or generated_suffix.strip() in ['nan', 'None', 'null']:
                    generated_suffix = "{none}"
                
                generated_suffix = generated_suffix.strip()

            return {"overlay_text": generated_suffix}
        
        except Exception as e:
            logger.critical(f"Handler error: {e}")
            return {"error": str(e), "overlay_text": "{none}"}