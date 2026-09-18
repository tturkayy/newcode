# newcode (ne-ural w-ave code)

i wanted to explore what happens when data encoding is treated less like a rigid grid of black-and-white cells and more like a continuous visual signal. traditional 2d barcodes rely heavily on high-contrast binary matrices; but honestly, i was just tired of seeing the same harsh, soulless black squares plastered everywhere. i wanted to build an aesthetic alternative, inspired by the elegant, continuous waveforms that silently carry data across the physical world in digital communications.

newcode takes inspiration from analog radio modulation to turn text into smooth optical waveforms, then uses a lightweight neural network to read them back directly in the browser. with a strict payload capacity of 39 bytes, newcode v1 technically outperforms both qr code version 1 and version 2 in raw byte limit, so take that, traditional squares. just don't ask it to encode an entire wikipedia article or run doom like that one qr code project ([kuberwastaken/backdooms](https://github.com/kuberwastaken/backdooms)).

| Format | Representation | Error Correction | Max Binary Payload (Bytes) |
| :--- | :--- | :--- | :---: |
| **QR Code Version 1** | 21x21 binary matrix | Level L (7%)<br>Level M (15%) | 17 B<br>14 B |
| **QR Code Version 2** | 25x25 binary matrix | Level L (7%)<br>Level M (15%) | 32 B<br>26 B |
| **NeWcode v1** | Continuous optical wave | RS(48, 40) GF(256) (~17%) | **39 B** |

it maps payload bytes through reed-solomon error correction and 16-apsk continuous modulation, extracting sub-pixel centroid features on the client and decoding them with a 1d cnn running on onnx runtime web (webgpu/wasm).

### live demo

generate wave patterns or decode an image directly in your browser:

> **[tturkayy.github.io/newcode/ ↗](https://tturkayy.github.io/newcode/)**

---

### tech stack

- **modulation:** continuous 16-apsk (pam-4 amplitude + qpsk phase)
- **error correction:** reed-solomon rs(48, 40) over gf(256)
- **demodulator:** 1d cnn running client-side via onnx runtime web
- **signal extraction:** sub-pixel weighted centroid slicing
