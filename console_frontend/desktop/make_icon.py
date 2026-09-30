"""Create the sharp, offline Windows icon for the local research workstation."""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter


root = Path(__file__).resolve().parent
size = 512
background = (5, 12, 8, 255)
green = (180, 255, 125, 255)
cyan = (100, 190, 166, 255)

icon = Image.new("RGBA", (size, size), background)
glow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
g = ImageDraw.Draw(glow)
g.rectangle((48, 48, 464, 464), outline=(98, 224, 110, 105), width=8)
g.polygon(((127, 126), (177, 126), (337, 338), (337, 386), (287, 386), (127, 174)), fill=(158, 255, 114, 150))
glow = glow.filter(ImageFilter.GaussianBlur(29))
icon = Image.alpha_composite(icon, glow)

d = ImageDraw.Draw(icon)
d.rectangle((41, 41, 471, 471), outline=(69, 133, 81, 255), width=4)
for x0, y0, x1, y1 in ((41, 41, 141, 41), (41, 41, 41, 141), (371, 41, 471, 41),
                       (471, 41, 471, 141), (41, 371, 41, 471), (41, 471, 141, 471),
                       (371, 471, 471, 471), (471, 371, 471, 471)):
    d.line((x0, y0, x1, y1), fill=green, width=7)

# Square terminal typography, drawn as geometry so it stays crisp at small sizes.
d.rectangle((126, 124, 174, 386), fill=green)
d.polygon(((174, 124), (218, 124), (337, 308), (337, 386), (292, 386), (174, 202)), fill=green)
d.rectangle((337, 124, 385, 386), fill=green)
d.rectangle((116, 401, 201, 408), fill=cyan)
d.rectangle((209, 401, 261, 408), fill=(69, 131, 86, 255))
d.rectangle((268, 401, 339, 408), fill=(69, 131, 86, 255))
d.rectangle((347, 401, 397, 408), fill=cyan)

icon.save(root / "icon.png")
icon.save(root / "icon.ico", format="ICO", sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)])
