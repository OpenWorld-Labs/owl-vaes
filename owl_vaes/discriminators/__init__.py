from .res import R3GANDiscriminator

def get_discriminator_cls(model_id):
    if model_id == "r3gan":
        return R3GANDiscriminator
    if model_id == "encodec":
        from .encodec import EncodecDiscriminator
        return EncodecDiscriminator
