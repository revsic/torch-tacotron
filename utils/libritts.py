import argparse

from speechset import AcousticDataset, Config
from speechset.datasets import LibriTTS
from speechset.utils import mp_dump, TextNormalizer, IDWrapper


@IDWrapper
class LibriTTSDataset(AcousticDataset):
    """LibriTTS dataset.
    """
    def __init__(self, data_dir: str, config: Config):
        """Initializer.
        Args:
            data_dir: path to the dataset.
            config: configuration.
        """
        super().__init__(LibriTTS(data_dir), config, TextNormalizer.REPORT_LOG)
    
    @classmethod
    def count_speakers(data_dir: str) -> int:
        """Count the number of the speakers.
        Args:
            data_dir: path to the dataset.
        Returns:
            the number of the speakers
        """
        return LibriTTS.count_speakers(data_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default=None)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--num-proc', default=4, type=int)
    args = parser.parse_args()

    config = Config(batch=None)
    libritts = LibriTTSDataset(args.data_dir, config)
    # dump
    mp_dump(libritts, args.output_dir, args.num_proc)
