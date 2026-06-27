"""Watermark overlay service."""
import io
import logging
from typing import Union, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont, ImageEnhance

logger = logging.getLogger(__name__)


class WatermarkService:
    """Service to apply text, logo, and tiled watermarks on images."""

    def __init__(self, default_opacity: float = 0.3):
        self.default_opacity = default_opacity

    def apply_text_watermark(
        self,
        image_input: Union[bytes, Image.Image],
        text: str,
        opacity: Optional[float] = None,
        font_size_ratio: float = 0.05,
        color: Tuple[int, int, int] = (255, 255, 255),
        position: str = "center"
    ) -> bytes:
        """
        Apply a simple text watermark to an image.
        
        Args:
            image_input: Raw image bytes or PIL Image object
            text: Watermark text (e.g. "© Smart Photo Sharing")
            opacity: Opacity between 0.0 (transparent) and 1.0 (opaque)
            font_size_ratio: Font size relative to image height
            color: RGB color tuple
            position: "center", "bottom-right", "bottom-left", "top-right", "top-left"
            
        Returns:
            Watermarked image bytes in JPEG format
        """
        img = self._get_image_object(image_input)
        img = img.convert("RGBA")
        
        # Create transparent overlay layer
        txt_layer = Image.Image()._new(img.im) if hasattr(img, 'im') else Image.new("RGBA", img.size, (255, 255, 255, 0))
        txt_layer = Image.new("RGBA", img.size, (255, 255, 255, 0))
        
        draw = ImageDraw.Draw(txt_layer)
        
        # Determine font size
        width, height = img.size
        font_size = max(16, int(height * font_size_ratio))
        
        try:
            # Fallback font handling
            font = ImageFont.load_default()
            # If standard font is available, use it (Pillow fallback behaves differently)
            # Standard truetype fonts aren't guaranteed on Windows/Linux by default,
            # so we start with default font.
        except Exception:
            font = ImageFont.load_default()

        # Calculate text bounding box/size
        try:
            text_w, text_h = draw.textsize(text, font=font)
        except AttributeError:
            # Pillow 10+ uses textbbox
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]

        # Determine coordinates
        x, y = self._get_position_coords(width, height, text_w, text_h, position)
        
        # Add opacity to color
        op = opacity if opacity is not None else self.default_opacity
        alpha = int(255 * op)
        rgba_color = color + (alpha,)
        
        # Draw text on overlay
        draw.text((x, y), text, fill=rgba_color, font=font)
        
        # Merge layers
        watermarked_img = Image.alpha_composite(img, txt_layer)
        return self._image_to_bytes(watermarked_img.convert("RGB"))

    def apply_tiled_watermark(
        self,
        image_input: Union[bytes, Image.Image],
        text: str,
        opacity: Optional[float] = None,
        font_size_ratio: float = 0.03,
        color: Tuple[int, int, int] = (255, 255, 255)
    ) -> bytes:
        """
        Apply a diagonal repeating tiled text watermark across the entire image.
        """
        img = self._get_image_object(image_input)
        img = img.convert("RGBA")
        
        # Create a drawing layer for text
        txt_layer = Image.new("RGBA", img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(txt_layer)
        
        width, height = img.size
        font_size = max(12, int(height * font_size_ratio))
        font = ImageFont.load_default()
        
        # Define color with opacity
        op = opacity if opacity is not None else self.default_opacity
        alpha = int(255 * op)
        rgba_color = color + (alpha,)
        
        # Tile spacing
        step_x = max(100, int(width / 4))
        step_y = max(100, int(height / 4))
        
        # Draw repeating text in a grid rotated diagonally
        for y in range(0, height, step_y):
            for x in range(0, width, step_x):
                # Apply slight offset to make grid look natural
                offset = (y // step_y) % 2 * (step_x // 2)
                draw.text((x + offset, y), text, fill=rgba_color, font=font)
                
        # Merge
        watermarked_img = Image.alpha_composite(img, txt_layer)
        return self._image_to_bytes(watermarked_img.convert("RGB"))

    def apply_logo_watermark(
        self,
        image_input: Union[bytes, Image.Image],
        logo_input: Union[bytes, Image.Image],
        opacity: Optional[float] = None,
        logo_size_ratio: float = 0.15,
        position: str = "bottom-right"
    ) -> bytes:
        """
        Apply an image logo watermark onto a photo.
        """
        img = self._get_image_object(image_input)
        logo = self._get_image_object(logo_input).convert("RGBA")
        
        img_w, img_h = img.size
        
        # Scale logo relative to main image size
        logo_w = int(img_w * logo_size_ratio)
        logo_h = int(logo.size[1] * (logo_w / logo.size[0]))
        logo = logo.resize((logo_w, logo_h), Image.Resampling.LANCZOS)
        
        # Adjust logo opacity/transparency
        op = opacity if opacity is not None else self.default_opacity
        if op < 1.0:
            alpha = logo.split()[3]
            alpha = ImageEnhance.Brightness(alpha).enhance(op)
            logo.putalpha(alpha)
            
        # Coordinates
        x, y = self._get_position_coords(img_w, img_h, logo_w, logo_h, position)
        
        # Paste logo on image
        img_rgba = img.convert("RGBA")
        logo_layer = Image.new("RGBA", img_rgba.size, (255, 255, 255, 0))
        logo_layer.paste(logo, (x, y))
        
        watermarked_img = Image.alpha_composite(img_rgba, logo_layer)
        return self._image_to_bytes(watermarked_img.convert("RGB"))

    # Helper private methods
    def _get_image_object(self, image_input: Union[bytes, Image.Image]) -> Image.Image:
        """Convert input to PIL Image object if it's in bytes."""
        if isinstance(image_input, bytes):
            return Image.open(io.BytesIO(image_input))
        return image_input

    def _image_to_bytes(self, img: Image.Image, format: str = "JPEG") -> bytes:
        """Convert PIL Image object back to raw bytes."""
        output = io.BytesIO()
        img.save(output, format=format, quality=85)
        return output.getvalue()

    def _get_position_coords(
        self, 
        img_w: int, 
        img_h: int, 
        watermark_w: int, 
        watermark_h: int, 
        position: str
    ) -> Tuple[int, int]:
        """Calculate X, Y coordinates for a watermark block based on position string."""
        margin = int(min(img_w, img_h) * 0.03)  # 3% margin
        
        if position == "bottom-right":
            return img_w - watermark_w - margin, img_h - watermark_h - margin
        elif position == "bottom-left":
            return margin, img_h - watermark_h - margin
        elif position == "top-right":
            return img_w - watermark_w - margin, margin
        elif position == "top-left":
            return margin, margin
        elif position == "center":
            return (img_w - watermark_w) // 2, (img_h - watermark_h) // 2
        else:
            # Default to bottom-right
            return img_w - watermark_w - margin, img_h - watermark_h - margin
