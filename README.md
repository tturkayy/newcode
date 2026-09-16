# newcode

i wanted to explore what happens when data encoding is treated less like a rigid grid of black-and-white cells and more like a continuous visual signal. traditional 2d barcodes rely heavily on high-contrast binary matrices; newcode takes inspiration from analog radio modulation to turn text into smooth optical waveforms, then uses a lightweight neural network to read them back directly in the browser.

it maps payload bytes through reed-solomon error correction and 16-apsk continuous modulation, extracting sub-pixel centroid features on the client and decoding them with a 1d cnn running on onnx runtime web (webgpu/wasm).

### live demo

generate wave patterns or decode an image directly in your browser:

👉 **[https://tturkayy.github.io/newcode/](https://tturkayy.github.io/newcode/)**

---

### tech stack

- **modulation:** continuous 16-apsk (pam-4 amplitude + qpsk phase)
- **error correction:** reed-solomon rs(48, 40) over gf(256)
- **demodulator:** 1d cnn running client-side via onnx runtime web
- **signal extraction:** sub-pixel weighted centroid slicing
