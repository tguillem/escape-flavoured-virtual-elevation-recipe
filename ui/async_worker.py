from PySide6.QtCore import QObject, Signal, Slot, QMutex, QMutexLocker

class AsyncWorker(QObject):
    trigger = Signal()
    resultReady = Signal(object)

    def __init__(self):
        super().__init__()
        self._mutex = QMutex()
        self._working = False
        self._new_values = False
        self._latest_values = {}
        self.trigger.connect(self.process)

    @Slot(str, float)
    def set_value(self, key: str, value: float):
        with QMutexLocker(self._mutex):
            self._latest_values[key] = value
            self._new_values = True
            if self._working:
                return
        self.trigger.emit()

    def set_values(self, values: dict):
        with QMutexLocker(self._mutex):
            self._latest_values = values.copy()
            self._new_values = True
            if self._working:
                return
        self.trigger.emit()

    @Slot()
    def process(self):
        self._mutex.lock()
        self._working = True

        while self._new_values:
            latest_values = self._latest_values.copy()
            self._new_values = False

            self._mutex.unlock()
            result = self._process_value(latest_values)
            self._mutex.lock()
            self.resultReady.emit(result)

        self._working = False
        self._mutex.unlock()

    def _process_value(self, value: dict):
        """
        Override this method in subclass.
        """
        raise NotImplementedError("Subclasses must implement _process_value")
