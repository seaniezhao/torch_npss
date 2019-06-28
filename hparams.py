import tensorflow as tf


def create_harmonic_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = tf.contrib.training.HParams(
        type=0,
        layers=3,
        blocks=2,
        dilation_channels=130,
        residual_channels=130,
        skip_channels=240,
        input_channel=60,
        condition_channel=343,
        output_channel=240,
        sample_channel=60,
        initial_kernel=10,
        kernel_size=2,
        bias=True
    )

    if hparams_string:
        tf.logging.info('Parsing harmonic hparams: %s', hparams_string)
        hparams.parse(hparams_string)

    if verbose:
        tf.logging.info('Final harmonic hparams: %s', hparams.values())

    return hparams


def create_aperiodic_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = tf.contrib.training.HParams(
        type=1,
        layers=3,
        blocks=2,
        dilation_channels=20,
        residual_channels=20,
        skip_channels=16,
        input_channel=64,
        condition_channel=358,
        output_channel=16,
        sample_channel=4,
        initial_kernel=10,
        kernel_size=2,
        bias=True
    )

    if hparams_string:
        tf.logging.info('Parsing aperiodic hparams: %s', hparams_string)
        hparams.parse(hparams_string)

    if verbose:
        tf.logging.info('Final aperiodic hparams: %s', hparams.values())

    return hparams


def create_vuv_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = tf.contrib.training.HParams(
        type=2,
        layers=3,
        blocks=2,
        dilation_channels=20,
        residual_channels=20,
        skip_channels=4,
        input_channel=65,
        condition_channel=358,
        output_channel=1,
        sample_channel=1,
        initial_kernel=10,
        kernel_size=2,
        bias=True
    )

    if hparams_string:
        tf.logging.info('Parsing vuv hparams: %s', hparams_string)
        hparams.parse(hparams_string)

    if verbose:
        tf.logging.info('Final vuv hparams: %s', hparams.values())

    return hparams


def create_f0_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = tf.contrib.training.HParams(
        type=3,
        layers=3,
        blocks=2,
        dilation_channels=130,
        residual_channels=130,
        skip_channels=240,
        input_channel=60,
        condition_channel=1126,
        cgm_factor=4,
        initial_kernel=10,
        kernel_size=2,
        bias=True
    )

    if hparams_string:
        tf.logging.info('Parsing f0 hparams: %s', hparams_string)
        hparams.parse(hparams_string)

    if verbose:
        tf.logging.info('f0 hparams: %s', hparams.values())

    return hparams


