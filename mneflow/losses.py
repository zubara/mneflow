# -*- coding: utf-8 -*-
"""
Custom Keras/TensorFlow loss functions for mneflow models: combinations of
cosine similarity, MSE/MAE, spectral (FFT-based), and Riemannian-distance
losses.

Created on Mon May  5 16:36:53 2025

@author: ipzub
"""
import tensorflow as tf


def Cos2MSE(y_true, y_pred, alpha=.5):
     """Sum of cosine similarity along axis 2 and MSE.

     Parameters
     ----------
     y_true : tf.Tensor
         Ground-truth values.

     y_pred : tf.Tensor
         Predicted values, same shape as ``y_true``.

     alpha : float, optional
         Unused. Defaults to 0.5. Present for interface consistency
         with the other loss functions in this module.

     Returns
     -------
     loss : tf.Tensor
         ``cosine_similarity(y_true, y_pred, axis=2)`` broadcast over
         a new trailing axis, plus ``mse(y_true, y_pred)``.

     """
     cosine = tf.keras.losses.cosine_similarity(y_true, y_pred, axis=[2])
     mse = tf.keras.losses.mse(y_true, y_pred)
     #mae = tf.keras.losses.mae(y_true, y_pred)
     #cosine2 = tf.keras.losses.cosine_similarity(y_true, y_pred, axis=[2])
     return cosine[:, :, :, tf.newaxis] + mse

def Cos2_3(y_true, y_pred, alpha=.5):
     """Sum of cosine similarity along axis 2 and along axis 3.

     Parameters
     ----------
     y_true : tf.Tensor
         Ground-truth values.

     y_pred : tf.Tensor
         Predicted values, same shape as ``y_true``.

     alpha : float, optional
         Unused. Defaults to 0.5. Present for interface consistency
         with the other loss functions in this module.

     Returns
     -------
     loss : tf.Tensor
         ``cosine_similarity(y_true, y_pred, axis=2)`` broadcast over
         a new trailing axis, plus ``cosine_similarity(y_true, y_pred,
         axis=3)`` broadcast over a new axis before the last.

     """
     cosine2 = tf.keras.losses.cosine_similarity(y_true, y_pred, axis=[2])
     #mse = tf.keras.losses.mse(y_true, y_pred)
     #mae = tf.keras.losses.mae(y_true, y_pred)
     cosine3 = tf.keras.losses.cosine_similarity(y_true, y_pred, axis=[3])
     return cosine2[:, :, :, tf.newaxis] + cosine3[:, :, tf.newaxis, :]

def Cos2MAE(y_true, y_pred, alpha=.5):
     """Sum of cosine similarity along axis 2 and MAE.

     Parameters
     ----------
     y_true : tf.Tensor
         Ground-truth values.

     y_pred : tf.Tensor
         Predicted values, same shape as ``y_true``.

     alpha : float, optional
         Unused. Defaults to 0.5. Present for interface consistency
         with the other loss functions in this module.

     Returns
     -------
     loss : tf.Tensor
         ``cosine_similarity(y_true, y_pred, axis=2)`` broadcast over
         a new trailing axis, plus ``mae(y_true, y_pred)``.

     """
     cosine = tf.keras.losses.cosine_similarity(y_true, y_pred, axis=[2])
     #mse = tf.keras.losses.mse(y_true, y_pred)
     mae = tf.keras.losses.mae(y_true, y_pred)
     #cosine2 = tf.keras.losses.cosine_similarity(y_true, y_pred, axis=[2])
     return cosine[:, :, :, tf.newaxis] + mae

def CosMSE(y_true, y_pred, alpha=.5):
     """Sum of cosine similarity along axis 3 and MSE.

     Parameters
     ----------
     y_true : tf.Tensor
         Ground-truth values.

     y_pred : tf.Tensor
         Predicted values, same shape as ``y_true``.

     alpha : float, optional
         Unused. Defaults to 0.5. Present for interface consistency
         with the other loss functions in this module.

     Returns
     -------
     loss : tf.Tensor
         ``cosine_similarity(y_true, y_pred, axis=3)`` broadcast over
         a new axis before the last, plus ``mse(y_true, y_pred)``.

     """
     cosine = tf.keras.losses.cosine_similarity(y_true, y_pred, axis=[3])
     mse = tf.keras.losses.mse(y_true, y_pred)
     #mae = tf.keras.losses.mae(y_true, y_pred)
     #cosine2 = tf.keras.losses.cosine_similarity(y_true, y_pred, axis=[2])
     return cosine[:, :, tf.newaxis, :] + mse

def Cos23MSE(y_true, y_pred, alpha=.1):
     """Weighted sum of cosine similarity along axes (2, 3) and MSE.

     Parameters
     ----------
     y_true : tf.Tensor
         Ground-truth values.

     y_pred : tf.Tensor
         Predicted values, same shape as ``y_true``.

     alpha : float, optional
         Weight given to the MSE term (``1 - alpha`` is given to the
         cosine term). Defaults to 0.1.

     Returns
     -------
     loss : tf.Tensor
         ``(1 - alpha) * cosine_similarity(y_true, y_pred, axis=[2, 3])``
         broadcast over two new trailing axes, plus
         ``alpha * mse(y_true, y_pred)``.

     """
     cosine = tf.keras.losses.cosine_similarity(y_true, y_pred, axis=[2, 3])
     mse = tf.keras.losses.mse(y_true, y_pred)
     #mae = tf.keras.losses.mae(y_true, y_pred)
     #cosine2 = tf.keras.losses.cosine_similarity(y_true, y_pred, axis=[2])
     return (1-alpha)*cosine[:, :, tf.newaxis, tf.newaxis] + alpha*mse

def Cos3MAE(y_true, y_pred, alpha=.5):
     """Sum of cosine similarity along axis 3 and MAE.

     Parameters
     ----------
     y_true : tf.Tensor
         Ground-truth values.

     y_pred : tf.Tensor
         Predicted values, same shape as ``y_true``.

     alpha : float, optional
         Unused. Defaults to 0.5. Present for interface consistency
         with the other loss functions in this module.

     Returns
     -------
     loss : tf.Tensor
         ``cosine_similarity(y_true, y_pred, axis=3)`` broadcast over
         a new axis before the last, plus ``mae(y_true, y_pred)``.

     """
     cosine = tf.keras.losses.cosine_similarity(y_true, y_pred, axis=[3])
     #mse = tf.keras.losses.mse(y_true, y_pred)
     mae = tf.keras.losses.mae(y_true, y_pred)
     #cosine2 = tf.keras.losses.cosine_similarity(y_true, y_pred, axis=[2])
     return cosine[:, :, tf.newaxis, :] + mae

def MSAE(y_true, y_pred, alpha=.5):
     """Weighted sum of MSE and MAE.

     Parameters
     ----------
     y_true : tf.Tensor
         Ground-truth values.

     y_pred : tf.Tensor
         Predicted values, same shape as ``y_true``.

     alpha : float, optional
         Weight given to the MSE term (``1 - alpha`` is given to the
         MAE term). Defaults to 0.5.

     Returns
     -------
     loss : tf.Tensor
         ``alpha * mse(y_true, y_pred) + (1 - alpha) * mae(y_true, y_pred)``.

     """
     #cosine = tf.keras.losses.cosine_similarity(y_true, y_pred, axis=[3])
     mse = tf.keras.losses.mse(y_true, y_pred)
     mae = tf.keras.losses.mae(y_true, y_pred)
     #cosine2 = tf.keras.losses.cosine_similarity(y_true, y_pred, axis=[2])
     return alpha*mse + (1-alpha)*mae

def dMSAE(y_true, y_pred, alpha=.5):
     """Elementwise maximum of MSE and MAE.

     Parameters
     ----------
     y_true : tf.Tensor
         Ground-truth values.

     y_pred : tf.Tensor
         Predicted values, same shape as ``y_true``.

     alpha : float, optional
         Unused. Defaults to 0.5. Present for interface consistency
         with the other loss functions in this module.

     Returns
     -------
     loss : tf.Tensor
         ``tf.maximum(mse(y_true, y_pred), mae(y_true, y_pred))``.

     """
     #cosine = tf.keras.losses.cosine_similarity(y_true, y_pred, axis=[3])
     mse = tf.keras.losses.mse(y_true, y_pred)
     mae = tf.keras.losses.mae(y_true, y_pred)
     #cosine2 = tf.keras.losses.cosine_similarity(y_true, y_pred, axis=[2])
     return tf.maximum(mse, mae)

# def top(y_true, y_pred, alpha=.5):
#      res = y_pred - y_true
#      return mse + mae

def spectral_loss_phase_magnitude(y_true, y_pred, mag_weight=1.0, phase_weight=0.5):
    """Spectral loss considering both magnitude and phase.

    Computes the FFT of ``y_true`` and ``y_pred`` and combines the
    mean squared error of their magnitudes with the mean squared
    error of their phases.

    Parameters
    ----------
    y_true : tf.Tensor
        Ground-truth values.

    y_pred : tf.Tensor
        Predicted values, same shape as ``y_true``.

    mag_weight : float, optional
        Weight applied to the magnitude loss term. Defaults to 1.0.

    phase_weight : float, optional
        Weight applied to the phase loss term. Defaults to 0.5.

    Returns
    -------
    loss : tf.Tensor
        Scalar weighted sum of the magnitude and phase losses.

    """
    # Compute FFT
    fft_true = tf.signal.fft(tf.cast(y_true, tf.complex64))
    fft_pred = tf.signal.fft(tf.cast(y_pred, tf.complex64))

    # Magnitude loss
    mag_true = tf.abs(fft_true)
    mag_pred = tf.abs(fft_pred)
    mag_loss = tf.reduce_mean(tf.square(mag_true - mag_pred))

    # Phase loss
    phase_true = tf.math.angle(fft_true)
    phase_pred = tf.math.angle(fft_pred)
    phase_loss = tf.reduce_mean(tf.square(phase_true - phase_pred))

    return mag_weight * mag_loss + phase_weight * phase_loss


def riemann_loss(y_true, y_pred):
    """
    Compute Riemannian distance for batches of positive definite matrices.

    Parameters
    ----------
    y_true : tf.Tensor
        Batch of positive definite matrices with shape
        ``[batch_size, n, n]``.

    y_pred : tf.Tensor
        Batch of positive definite matrices with shape
        ``[batch_size, n, n]``.

    Returns
    -------
    distances : tf.Tensor
        Tensor of shape ``[batch_size]`` containing the Riemannian
        distance between each pair of matrices in ``y_true`` and
        ``y_pred``.

    """
    # Map the riemannian_distance function over the batch dimension
    distances = tf.map_fn(
        lambda x: riemann_distance(x[1], x[0]),
        (y_true, y_pred),
        fn_output_signature=tf.float32
    )

    return distances


    return


def tensor_covariance(tensor, axis=2):
    """
    Compute the covariance of a 4D tensor along a specified axis.

    Parameters
    ----------
    tensor : tf.Tensor
        4D tensor of shape ``[dim0, dim1, dim2, dim3]``.

    axis : int, optional
        The axis along which to compute covariance (0, 1, 2, or 3).
        Defaults to 2.

    Returns
    -------
    cov : tf.Tensor
        Covariance matrix computed by flattening the remaining axes
        and taking ``X^T X / n`` along ``axis``.

    """
    # Get tensor shape
    shape = tensor.shape
    #print("tensor_covariance input shape:", shape)

    # Center the data along the specified axis
    #mean = tf.reduce_mean(tensor, axis=-2, keepdims=True)
    centered = tensor #- mean

    # Reshape the tensor for covariance calculation
    if axis == 0:
        # Computing covariance across dim0
        # Result shape will be [dim1*dim2*dim3, dim1*dim2*dim3]
        reshaped = tf.reshape(centered, [shape[0], -1])  # [dim0, dim1*dim2*dim3]
        # Transpose to [dim1*dim2*dim3, dim0]
        #reshaped = tf.transpose(reshaped)

    elif axis == 1:
        # Computing covariance across dim1
        # Result shape will be [dim0*dim2*dim3, dim0*dim2*dim3]
        # Permute to [dim1, dim0, dim2, dim3]
        permuted = tf.transpose(centered, [1, 0, 2])
        reshaped = tf.reshape(permuted, [shape[1], -1])  # [dim1, dim0*dim2*dim3]
        # Transpose to [dim0*dim2*dim3, dim1]
        reshaped = tf.transpose(reshaped)

    elif axis == 2:
        # Computing covariance across dim2
        # Result shape will be [dim0*dim1*dim3, dim0*dim1*dim3]
        # Permute to [dim2, dim0, dim1, dim3]
        permuted = tf.transpose(centered, [2, 0, 1])
        reshaped = tf.reshape(permuted, [shape[2], -1])  # [dim2, dim0*dim1*dim3]
        # Transpose to [dim0*dim1*dim3, dim2]
        reshaped = tf.transpose(reshaped)

    elif axis == 3:
        # Computing covariance across dim3
        # Result shape will be [dim0*dim1*dim2, dim0*dim1*dim2]
        # Permute to [dim3, dim0, dim1, dim2]
        permuted = tf.transpose(centered, [2, 0, 1])
        reshaped = tf.reshape(permuted, [shape[3], -1])  # [dim3, dim0*dim1*dim2]
        # Transpose to [dim0*dim1*dim2, dim3]
        reshaped = tf.transpose(reshaped)

    else:
        raise ValueError("Axis must be 0, 1, 2, or 3 for a 4D tensor")

    # Compute covariance: 1/N * X^T * X
    n = tf.cast(tf.shape(reshaped)[0], tf.float32)
    cov = tf.matmul(reshaped, reshaped, transpose_a=True) / n
    #print("tensor_covariance output shape:", cov.shape)
    assert cov.shape[0] == tensor.shape[axis]

    return cov





def riemann_distance(y_true, y_pred, axis=2):
    """
    Calculate Riemannian distance between two tensors along the axis.

    Parameters
    ----------
    y_true : tf.Tensor
        Positive definite matrix (or batch of matrices), with
        covariance computed along ``axis``.

    y_pred : tf.Tensor
        Positive definite matrix (or batch of matrices) of the same
        dimensions as ``y_true``.

    axis : int, optional
        Axis along which :func:`tensor_covariance` computes the
        covariance of ``y_true`` and ``y_pred``. Defaults to 2. Note
        that the covariance calls below currently pass ``axis=2``
        explicitly regardless of this argument.

    Returns
    -------
    distance : tf.Tensor
        The Riemannian distance(s) between ``y_true`` and ``y_pred``.

    """
    # For numerical stability we use the formulation:
    # d(A,B) = ||log(A^(-1/2) B A^(-1/2))||_F
    #cov_A = tf.matmul(A, A)
    # Compute the eigendecomposition of A
    cov_true = tensor_covariance(y_true, axis=2)
    cov_pred = tensor_covariance(y_pred, axis=2)

    eigenvalues_A, eigenvectors_A = tf.linalg.eigh(cov_true)

    # Avoid numerical issues with very small eigenvalues
    eps = 1e-10
    eigenvalues_A = tf.maximum(eigenvalues_A, eps)

    # Compute A^(-1/2)
    sqrt_inv_eigenvalues = tf.pow(eigenvalues_A, -0.5)
    # Create a diagonal matrix from the sqrt_inv_eigenvalues
    sqrt_inv_eigenvalues_diag = tf.linalg.diag(sqrt_inv_eigenvalues)

    # A^(-1/2) = U * D^(-1/2) * U^T where U are eigenvectors and D are eigenvalues
    A_sqrt_inv = tf.matmul(
        tf.matmul(eigenvectors_A, sqrt_inv_eigenvalues_diag),
        eigenvectors_A, transpose_b=True
    )

    # Compute A^(-1/2) B A^(-1/2)
    C = tf.matmul(tf.matmul(A_sqrt_inv, cov_pred), A_sqrt_inv)

    # Compute the eigendecomposition of C
    eigenvalues_C, _ = tf.linalg.eigh(C)
    eigenvalues_C = tf.maximum(eigenvalues_C, eps)  # Ensure positive eigenvalues

    # The Riemannian distance is the Frobenius norm of log(C)
    # For a positive definite matrix, the Frobenius norm of log(C) is equivalent to
    # the L2 norm of the log of its eigenvalues
    log_eigenvalues = tf.math.log(eigenvalues_C)

    # Compute the Frobenius norm (L2 norm of eigenvalues)
    distance = tf.sqrt(tf.reduce_sum(tf.square(log_eigenvalues), axis=-1))

    return distance
