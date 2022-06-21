import time
import typing
from collections import deque

from PIL import ImageGrab
import pyautogui
import cv2
import numpy as np
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
    # noinspection PyTypeChecker
    image = np.array(ImageGrab.grab(bbox))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


# noinspection PyMissingConstructor
class WindowCapture(cv2.VideoCapture):
    def __init__(self, win_or_name: typing.Union[str, Window]):
        if isinstance(win_or_name, str):
            win_or_name = pyautogui.getWindowsWithTitle(win_or_name)[0]
        self.window = win_or_name

    def read(self, _=None) -> tuple[bool, np.ndarray]:
        x, y, w, h = self.window.box
        return True, get_screen_image(BoundingBox(x, y, x + w, y + h))

    def __repr__(self) -> str:
        return repr(self.window.box)


def process_capture() -> None:
    cap = WindowCapture(pyautogui.getActiveWindow())
    fps_meter = FpsMeter()

    while True:
        has_read, image = cap.read()

        fps_meter.tick()
        fps_text = f'FPS={fps_meter.fps:02.2f} DELAY={fps_meter.delay:.2f}'
        draw_text(image, (50, 50), fps_text)

        cv2.imshow('image', image)

        key = cv2.waitKey(1)
        if key in (27, ord('q'), ord('Q')):
            cv2.destroyAllWindows()
            raise KeyboardInterrupt()


def main():
    try:
        process_capture()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
