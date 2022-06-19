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


class FpsMeter:
    def __init__(self):
        self.times = deque()

    def tick(self, ) -> None:
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


def get_screen_image(bbox: tuple[int, ...] = None) -> np.ndarray:
    # noinspection PyTypeChecker
    image = np.array(ImageGrab.grab(bbox))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


class WindowCapture:
    def __init__(self, win_or_name: typing.Union[Window, str]):
        if isinstance(win_or_name, str):
            self.window = pyautogui.getWindowsWithTitle(win_or_name)[0]
        else:
            self.window = win_or_name

    def get_image(self) -> np.ndarray:
        return get_screen_image(self.window.box)

    def __repr__(self) -> str:
        return repr(self.window.box)


def main():
    cap = WindowCapture(pyautogui.getActiveWindow())
    fps_meter = FpsMeter()

    while True:
        image = cap.get_image()

        fps_meter.tick()
        fps_text = f'FPS={fps_meter.fps:02.2f} DELAY={fps_meter.delay:.2f}'
        draw_text(image, (50, 50), fps_text)

        cv2.imshow('image', image)

        key = cv2.waitKey(1)
        if key in (27, ord('q'), ord('Q')):
            cv2.destroyAllWindows()
            raise KeyboardInterrupt()


def main2():
    window = pyautogui.getActiveWindow()
    print(repr(window.box))
    print(*window.box)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
