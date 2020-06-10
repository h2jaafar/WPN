from simulator.services.resources.directories import ModelDir, CacheDir, ImagesDir, MapsDir, ScreenshotsDir, \
    TrainingDataDir
from simulator.services.resources.directory import Directory
from simulator.services.services import Services


class Resources(Directory):
    model_dir: ModelDir
    cache_dir: CacheDir
    images_dir: ImagesDir
    maps_dir: MapsDir
    screenshots_dir: ScreenshotsDir
    training_data_dir: TrainingDataDir

    def __init__(self, services: Services):
        super().__init__(services, "src/resources", "./")

        self.model_dir = ModelDir(self._services, "algorithms", self._full_path())
        self.cache_dir = CacheDir(self._services, "cache", self._full_path())
        self.images_dir = ImagesDir(self._services, "images", self._full_path())
        self.maps_dir = MapsDir(self._services, "maps", self._full_path())
        self.screenshots_dir = ScreenshotsDir(self._services, "screenshots", self._full_path())
        self.training_data_dir = TrainingDataDir(self._services, "training_data", self._full_path())
