import os
import time
import typing
from collections import deque

import cv2
import numpy as np
import pyautogui
import pyrect
from PIL import ImageGrab, Image, ImageFont, ImageDraw
from pywinctl import Window


def draw_text(
    frame: np.ndarray,
    point: tuple[int, int],
    text: str,
    *,
    color: tuple[int, int, int] = (0, 0, 255),
    font_scale: float = 1.0,
    thickness: int = 1,
    font: int = cv2.FONT_HERSHEY_COMPLEX_SMALL,
) -> None:
    """
    Добавляет на изображение текст в выбранной позиции (без копирования).
    """
    cv2.putText(frame, text, point, font, font_scale, color, thickness, cv2.LINE_AA)


class BoundingBox(typing.NamedTuple):
    x1: int
    y1: int
    x2: int
    y2: int


class FpsMeter:
    def __init__(self):
        self.times = deque()

    def tick(self) -> None:
        now = time.perf_counter()

        self.times.append(now)
        if len(self.times) > 2:
            self.times.popleft()

    @property
    def fps(self) -> float:
        return 1 / self.delay

    @property
    def delay(self) -> float:
        if len(self.times) >= 2:
            return self.times[-1] - self.times[-2]
        return float('nan')


def get_screen_image(bbox: BoundingBox = None) -> np.ndarray:
    image = cv2.cvtColor(np.array(ImageGrab.grab(bbox)), cv2.COLOR_RGB2BGR)
    return image


class WindowCapture(cv2.VideoCapture):
    # noinspection PyMissingConstructor
    def __init__(self, window_or_name: typing.Union[str, Window]):
        if isinstance(window_or_name, str):
            window_or_name = pyautogui.getWindowsWithTitle(window_or_name)[0]
        self._window = window_or_name

    def read(self, _=None) -> tuple[bool, np.ndarray]:
        x, y, w, h = self._window.box
        return True, get_screen_image(BoundingBox(x, y, x + w, y + h))

    def window_box(self) -> pyrect.Box:
        return self._window.box

    def __repr__(self) -> str:
        return (
            f'<object {self.__class__.__name__}('
            f'title={self._window.title!r} '
            f'box={self._window.box!r}'
            f')>'
        )


def draw_fps(image: np.ndarray, meter: FpsMeter) -> None:
    meter.tick()
    fps_text = f'FPS={meter.fps:.2f}\nDELAY={meter.delay:.2f}'

    font_path = os.path.join(cv2.__path__[0], 'qt', 'fonts', 'DejaVuSans.ttf')
    font = ImageFont.truetype(font_path, size=80)

    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    draw.text((0, 0), fps_text, fill='red', anchor='la', font=font)

    image[:, :, :] = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)


def process_capture() -> None:
    fps_meter = FpsMeter()
    cap = WindowCapture(pyautogui.getActiveWindow())

    sizer = 0.4
    *_, width, height = cap.window_box()
    window_name = 'image'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(window_name, int(width * sizer), int(height * sizer))

    while True:
        has_read, image = cap.read()

        draw_fps(image, fps_meter)
        cv2.imshow(window_name, image)

        key = cv2.waitKey(1)
        if key in (27, ord('q'), ord('Q')):
            return
        if key in (ord(' '), ord('p'), ord('P')):
            cv2.waitKey()


def main() -> None:
    try:
        process_capture()
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
