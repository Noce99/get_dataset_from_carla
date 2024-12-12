import torch
from abc import ABC, abstractmethod
from .events_visualizations import voxel_grid_stereo_to_rgb, voxel_grid_mono_to_rgb, \
                                   histogram_mono_to_rgb,histogram_stereo_to_rgb


class EventRepresentation(ABC):

    def __init__(self, channels):
        self.channels = channels

    @abstractmethod
    def convert(self, x: torch.Tensor, y: torch.Tensor, pol: torch.Tensor, time: torch.Tensor):
        pass

    @abstractmethod
    def get_dataset_file_name(self, delta_t_ms, train_validation):
        pass

    @abstractmethod
    def to_rgb_stereo(self, representation_left, representation_right):
        pass

    @abstractmethod
    def to_rgb_mono(self, representation):
        pass

class Histogram(EventRepresentation):

    def __init__(self, height: int, width: int, normalize: bool):
        super().__init__(2)
        self.height = height
        self.width = width
        self.normalize = normalize

    @classmethod
    def from_configuration(cls, configuration):
        assert configuration["representation_type"] == "histogram"
        return cls(height=int(configuration["height"]),
                   width=int(configuration["width"]),
                   normalize=bool(configuration["normalize"]))

    def convert(self, x: torch.Tensor, y: torch.Tensor, pol: torch.Tensor, time: torch.Tensor):
        assert x.shape == y.shape == pol.shape == time.shape
        assert x.ndim == 1

        pol = (pol.float()/2 + 0.5).int()

        print(f"X in [{x.min()}; {x.max()}] [{self.width}]")
        print(f"Y in [{y.min()}; {y.max()}] [{self.height}]")
        print(f"POL in [{torch.unique(pol)}]")

        histo = torch.zeros((2, self.height, self.width), dtype=torch.float, requires_grad=False)
        with (torch.no_grad()):
            pol = pol.int() # Let's make the polarity an integer

            x0 = x.int()  # Let's make the x an integer
            y0 = y.int()  # Let's make the y an integer


            errors = True
            while errors:
                try:
                    for xlim in [x0, x0 + 1]:
                        for ylim in [y0, y0 + 1]:
                            interp_weights = (1 - (xlim - x).abs()) * (1 - (ylim - y).abs()).float()
                            index = (pol*self.height*self.width + self.width * ylim.long() + xlim.long()).long()

                            mask = (xlim < self.width) & (xlim >= 0) & (ylim < self.height) & (ylim >= 0) & (index > 0) \
                                & (index < 2*self.height*self.width)

                            histo.put_(index[mask], interp_weights[mask], accumulate=True)
                    errors = False
                except IndexError as e:
                    print("I saved you from an index error.")
                    print("@"*30)
                    print(e)
                    print("@" * 30)
                    print(f"interp_weights in [{interp_weights.min()}; {interp_weights.max()}]")
                    print(f"index in [{index.min()}; {index.max()}]")
                    print(f"x in [{x.min()}; {x.max()}]")
                    print(f"x0 in [{x0.min()}; {x0.max()}]")
                    print(f"y in [{y.min()}; {y.max()}]")
                    print(f"y0 in [{y0.min()}; {y0.max()}]")
                    print(f"pol in [{pol.min()}; {pol.max()}]")

            if self.normalize:
                mask = torch.nonzero(histo, as_tuple=True)
                if mask[0].size()[0] > 0:
                    mean = histo[mask].mean()
                    std = histo[mask].std()
                    if std > 0:
                        histo[mask] = (histo[mask] - mean) / std
                    else:
                        histo[mask] = histo[mask] - mean

        return histo

    def get_dataset_file_name(self, delta_t_ms, train_validation):
        if train_validation == "train":
            dataset_name = f"TRAIN_histogram_{delta_t_ms}_hdf5"
        else:
            dataset_name = f"VALIDATION_histogram_{delta_t_ms}_hdf5"
        return dataset_name

    def to_rgb_stereo(self, representation_left, representation_right):
        return histogram_stereo_to_rgb(representation_left, representation_right)

    def to_rgb_mono(self, representation):
        return histogram_mono_to_rgb(representation)
