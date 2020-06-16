import numpy as np
from abc import ABC, abstractmethod


class Loss(ABC):
    '''
    Abstract class representing a loss function
    '''
    @abstractmethod
    def get_loss(self, scores, labels):
        """
        will return the loss value
        """


class SoftmaxCrossEntropyLoss(Loss):
    '''
    Represents the categorical softmax cross entropy loss
    '''

    def get_loss(self, scores, labels):
        '''
        Calculates the average categorical softmax cross entropy loss.

        Args:
            scores (numpy.ndarray): Unnormalized logit class scores.
            labels (numpy.ndarray): True labels represented as ints
        Returns:
            loss (float): The average cross entropy
            grad (numpy.ndarray): Gradient for scores w.r.t the loss.
        '''
        scores_norm = scores - np.max(scores, axis=1, keepdims=True)
        scores_norm = np.exp(scores_norm)
        scores_norm = scores_norm / np.sum(scores_norm, axis=1, keepdims=True)

        true_class_scores = scores_norm[np.arange(len(labels)), labels]
        loss = np.mean(-np.log(true_class_scores))

        one_hot = np.zeros(scores.shape)
        one_hot[np.arange(len(labels)), labels] = 1.0
        grad = (scores_norm - one_hot) / len(labels)

        return loss, grad
