import os
import shutil
from typing import TYPE_CHECKING, Any, Callable

import pygame
import torch
import matplotlib.pyplot as plt

from simulator.services.debug import DebugLevel
from simulator.services.resources.atlas import Atlas
from simulator.services.resources.directory import Directory
from simulator.services.services import Services

if TYPE_CHECKING:
    from algorithms.lstm.ML_model import MLModel


class ModelSubdir(Directory):
    @staticmethod
    def _model_default_save(dir: 'Directory', name: str, model: 'MLModel'):
        Directory._default_save(dir, name + "_config", (model.config, type(model)))
        torch.save(model.state_dict(), dir._full_path() + name)

    @staticmethod
    def _model_default_load(dir: 'Directory', name: str) -> 'MLModel':
        config, model_type = super().load(name + "_config")
        model = model_type(dir._services, config)
        model.load_state_dict(torch.load(dir._full_path() + name))
        model.eval()
        return model

    def save(self, name: str, obj: 'MLModel',
             save_function: Callable[['Directory', str, 'MLModel'], None] = None) -> None:
        if save_function:
            raise NotImplementedError()

        super().save(name, obj.to(self._services.torch.cpu), ModelSubdir._model_default_save)
        obj.to(self._services.torch.device)

    def load(self, name: str,
             load_function: Callable[['Directory', str], 'MLModel'] = None) -> 'MLModel':
        if load_function:
            raise NotImplementedError()

        model = super().load(name, ModelSubdir._model_default_load)
        if model:
            model = model.to(self._services.torch.device)
        return model

    def save_figure(self, plot_name: str) -> None:
        plt.savefig(self._full_path() + self.name_without_trailing_slash() + "_" + plot_name)


class ModelDir(Directory):
    def save(self, name: str, obj: 'MLModel',
             save_function: Callable[['Directory', str, 'MLModel'], None] = None) -> None:
        if save_function:
            raise NotImplementedError()

        model_subdir = ModelSubdir(self._services, name, self._full_path(), create=True, overwrite=True)
        model_subdir.save(name, obj)

    def load(self, name: str,
             load_function: Callable[['Directory', str], 'MLModel'] = None) -> 'MLModel':
        if load_function:
            raise NotImplementedError()

        model_subdir = self.get_subdir(name)
        return model_subdir.load(name)

    def get_subdir(self, name: str) -> ModelSubdir:
        return ModelSubdir(self._services, name, self._full_path())


class CacheDir(Directory):
    def clear(self) -> None:
        try:
            shutil.rmtree(self._full_path())
        except FileNotFoundError:
            pass

        os.mkdir(self._full_path())
        self._services.debug.write("Cache cleared", DebugLevel.BASIC)

    def delete_entry(self, cache_name: str):
        cache_name = self._add_extension(cache_name, "pickle")
        try:
            os.remove(self._full_path() + cache_name)
        except FileNotFoundError:
            pass

    def get_or_save(self, cache_name: str, func: Callable[[], Any],
                    save_function: Callable[['Directory', str, Any], None] = None,
                    load_function: Callable[['Directory', str], Any] = None) -> Any:
        obj: Any = self.load(cache_name, load_function)
        if not obj:
            obj = func()
            self.save(cache_name, obj, save_function)
        return obj


class ImagesDir(Atlas):
    @staticmethod
    def _image_default_save(dir: Directory, name: str, image: pygame.Surface) -> None:
        name = dir._add_extension(name, "png")
        pygame.image.save(image, dir._full_path() + name)

    @staticmethod
    def _image_default_load(dir: Directory, name: str) -> pygame.Surface:
        name = dir._add_extension(name, "png")
        image = pygame.image.load(dir._full_path() + name)
        return image

    def save(self, name: str, obj: pygame.Surface,
             save_function: Callable[['Directory', str, pygame.Surface], None] = None) -> None:
        super().save(name, obj, ImagesDir._image_default_save)

    def load(self, name: str,
             load_function: Callable[['Directory', str], pygame.Surface] = None) -> pygame.Surface:
        return super().load(name, ImagesDir._image_default_load)

    def create_atlas(self, atlas_name: str) -> 'ImagesDir':
        return ImagesDir(self._services, atlas_name, self._full_path(), True)

    def get_atlas(self, atlas_name: str) -> 'ImagesDir':
        return ImagesDir(self._services, atlas_name, self._full_path())


class MapsDir(Directory):
    def create_atlas(self, atlas_name: str) -> Atlas:
        return Atlas(self._services, atlas_name, self._full_path(), True)

    def get_atlas(self, atlas_name: str) -> Atlas:
        return Atlas(self._services, atlas_name, self._full_path())


class ScreenshotsDir(ImagesDir):
    def __init__(self, services: Services, name: str, parent: str):
        super().__init__(services, name, parent)

    def append(self, screenshot: pygame.Surface) -> None:
        file_name: str = "screenshot_" + str(self._get_next_index())
        self.save(file_name, screenshot)
        self._increment_index()


class TrainingDataDir(Directory):
    pass
