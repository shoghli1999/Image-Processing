# Edge detection: five classic methods compared

A bachelor's course project in digital image processing. I implemented five edge detectors with OpenCV and compared them on the same image:

- Canny
- Sobel
- Prewitt
- Roberts
- Laplacian

`all_together.py` asks for an image path and shows the original next to all five results in one figure. Each method also has its own short script.

```bash
pip install -r requirements.txt
python all_together.py
python canny.py     # or sobel.py, prewitt.py, roberts.py, lap.py
```

What I saw: Canny gives the cleanest, thinnest edges but needs the most computation. Sobel and Prewitt are close to each other and handle noise reasonably well. Roberts is fast but noisy. The Laplacian finds fine detail in every direction but reacts strongly to noise and can give double edges.

`final.pdf` and `final.pptx` are the report and the slides I presented.
