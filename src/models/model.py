from abc import ABCMeta, abstractmethod


class Model(metaclass=ABCMeta):
    @abstractmethod
    def show_sidebar(self):
        raise NotImplementedError()

    @abstractmethod
    def show_page(self):
        raise NotImplementedError()
