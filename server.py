import asyncio
import json
import logging
import os

import aiohttp_cors
import cv2
from aiohttp import web
from av import VideoFrame

from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay

from eaudio import apply_audio_effects

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()
relay = MediaRelay()


class AudioTransformTrack(MediaStreamTrack):
    kind = 'audio'

    def __init__(self, track, audio_effect):
        super().__init__()
        self.track = track
        self.audio_effect = audio_effect

    async def recv(self):
        frame = await self.track.recv()
        return await apply_audio_effects(audio_effect=self.audio_effect, frame=frame)


class VideoTransformTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    """

    kind = "video"

    def __init__(self, track, transform):
        super().__init__()  # don't forget this!
        self.track = track
        self.transform = transform

    async def recv(self):
        frame = await self.track.recv()

        if self.transform == "cartoon":
            img = frame.to_ndarray(format="bgr24")

            # prepare color
            img_color = cv2.pyrDown(cv2.pyrDown(img))
            for _ in range(6):
                img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
            img_color = cv2.pyrUp(cv2.pyrUp(img_color))

            # prepare edges
            img_edges = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img_edges = cv2.adaptiveThreshold(
                cv2.medianBlur(img_edges, 7),
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                9,
                2,
            )
            img_edges = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2RGB)

            # combine color and edges
            img = cv2.bitwise_and(img_color, img_edges)

            # rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame
        elif self.transform == "edges":
            # perform edge detection
            img = frame.to_ndarray(format="bgr24")
            img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)

            # rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame
        elif self.transform == "rotate":
            # rotate image
            img = frame.to_ndarray(format="bgr24")
            rows, cols, _ = img.shape
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), frame.time * 45, 1)
            img = cv2.warpAffine(img, M, (cols, rows))

            # rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame
        else:
            img = frame.to_ndarray(format="bgr24")

            # ===================== Custom Image ======================
            # font
            font = cv2.FONT_HERSHEY_SIMPLEX
            # org
            height, width, channels = img.shape
            org = (int(width / 2), int(height / 2))
            # fontScale
            fontScale = 1
            # Blue color in BGR
            color = (200, 0, 0)
            # Line thickness of 2 px
            thickness = 2
            # Using cv2.putText() method
            img = cv2.putText(img, self.transform, org, font, fontScale, color, thickness, cv2.LINE_AA)

            # Test coordinates to draw a line
            x, y, w, h = 108, 107, 193, 204

            # Draw line on overlay and original input image to show difference
            cv2.line(img, (x, y), (x + w, x + h), (36, 255, 12), 6)
            cv2.line(img, (x, y), (x + w, x + h), (36, 255, 12), 6)

            # Transparency value
            alpha = 0.50

            # Perform weighted addition of the input image and the overlay
            img = cv2.addWeighted(img, alpha, img, 1 - alpha, 0)

            ksize = (10, 10)

            # Using cv2.blur() method
            img = cv2.blur(img, ksize)
            # ===================== Custom Image ======================

            # rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame


async def index(request):
    content = open(os.path.join(ROOT, "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


async def javascript(request):
    content = open(os.path.join(ROOT, "client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    print("Created for %s", request.remote)

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            if isinstance(message, str) and message.startswith("ping"):
                channel.send("pong" + message[4:])

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print("Connection state is %s", pc.connectionState)

        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        print("Track %s received", track.kind)

        if track.kind == "video":
            pc.addTrack(VideoTransformTrack(relay.subscribe(track), transform=params["video"]))
        elif track.kind == "audio":
            pc.addTrack(AudioTransformTrack(track, audio_effect=params['audio']))

    # handle offer
    await pc.setRemoteDescription(offer)

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    app = web.Application()
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
        )
    })

    app.on_shutdown.append(on_shutdown)
    cors.add(app.router.add_get("/", index))
    cors.add(app.router.add_get("/client.js", javascript))
    cors.add(app.router.add_post("/offer", offer))

    web.run_app(app, access_log=None, host="127.0.0.1", port=8080)
