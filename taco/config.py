class Config:
    """Model configuration.
    """
    def __init__(self, vocabs: int, mel: int):
        """Initializer.
        Args:
            vocabs: the size of the dictionary.
            mel: the number of the mel-scale filterbank bins.
        """
        self.vocabs = vocabs
        self.mel = mel

        # channel info
        self.embeddings = 256
        self.channels = 256

        # encoder
        self.enc_prenet = [256]
        self.enc_dropout = 0.5

        # cbhg
        self.cbhg_banks = 16
        self.cbhg_pool = 2
        self.cbhg_kernels = 3
        self.cbhg_highways = 4

        # reduction
        self.reduction = 2

        # align
        self.fwd_loc = 32
        self.fwd_kernels = 11

        # decoder
        self.dec_prenet = [80, 128]
        self.dec_dropout = 0.
        self.dec_layers = 2
        
        # teacher force
        self.teacher_force = None

        # inference
        self.dec_max_factor = 8 // self.reduction
