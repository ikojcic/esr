import numpy as np

class SourceSimulator:

    def __init__(self, model, sources = 0, waveform = None):
        """
        Parent class for source simulators.
        Args:
             model: An instance of class Model.

             sources (optional): Indices of active sources (row(s) in the
             matrix of source amplitudes). It must be an integer, 1D numpy
             array of integers, or a list of integers.

             waveform (optional): Desired waveform of the source activity
             simulation signal. Must be a sequence convertible to a 1D numpy
             array of floats.

        Raises:
             ValueError: If sources are not: an integer, 1D numpy array of
             integers or a list of integers.
                         If sources are a numpy array that is not 1D.
                         If waveform is a numpy array that is not 1D.
             TypeError: If waveform is not convertible to a numpy array of
             floats.
             NotImplementedError: If simulate method is not implemented in
             the child class.
        """

        self.model = model
        self.nb_sources = np.asarray(self.model.cortex.nb_vertices)

        if isinstance(sources, int):
            sources = [sources]

        elif isinstance(sources, np.ndarray):
            if sources.ndim != 1:
                raise ValueError('\'sources\' must have a dimension 1, '
                                 ' not {}. '.format(sources.ndim))
            if (sources.dtype ==int) == False:
                raise ValueError('\'sources\' must be a 1D numpy array of '
                                'integers')

            sources = sources.tolist()

        elif isinstance(sources, list):
            if all([isinstance(src_id, int) for src_id in sources]) == False:
                raise ValueError('\'sources\' must be a list of integers.')

        else:
            raise ValueError('\'sources\' must be an integer.')


        if waveform is None:
            waveform = np.sin(np.linspace(0, 2*np.pi, 1000))
        elif isinstance(waveform, np.ndarray) and waveform.ndim != 1:
                raise ValueError('\'waveform\' must have a dimension 1, '
                                 ' not {}. '.format(waveform.ndim))
        elif isinstance(waveform, np.ndarray) == False:
            try:
                waveform = np.array(waveform, dtype = np.float64)
            except Exception:
                raise TypeError('\'waveform\' must be convertible to a numpy '
                                'array of floats.')

        self.sources = sources
        self.waveform = waveform

    def description(self):
        return "Cortex: {}\n Forward operator: {}".format(self.model.cortex,
                                                       self.model.forward)

    def simulate(self):
        raise NotImplementedError


class PointSourceSimulator(SourceSimulator):

    def simulate(self):
        """Point source simulator."""
        
        T = len(self.waveform)
        X = np.zeros((self.nb_sources, T))
        for source_id in self.sources:
            X[source_id, :] = self.waveform
        return X


class SpreadSourceSimulator(SourceSimulator):

    def __init__(self, model, sources = 0, waveform = None, decay= None):
        """
        Spread source simulator.
        Args: decay (optional): Spread weights. Describes the spreading of
             source activity to its neighbours. It must be a list or a 1D
             numpy array of floats in descending order.
        Raises:
            ValueError: If decay is a numpy array which is not 1D.
                        If decay is not in descending order.
        """

        if decay is None:
            decay = np.array([1, 0.7, 0.3])
        elif isinstance(decay, np.ndarray):
            if decay.ndim != 1:
                raise ValueError('\'decay\' must have a dimension 1, '
                                  ' not {}. '.format(decay.ndim))
            if np.any(np.diff(decay)>=0):
                raise ValueError('\'decay\' must be in descending order')

        elif isinstance(decay, list):
            if all(decay[i] >= decay[i + 1] for i in range(len(decay) - 1)):
                decay = np.array(decay, dtype = np.float64)
            else:
                raise ValueError('\'decay\' must be in descending order')

        super().__init__(model, sources, waveform)
        self.decay = decay
        self.tri = self.model.cortex.triangles


    def simulate(self):
        T = len(self.waveform)
        X = np.zeros((self.nb_sources, T))
        for source1_id in self.sources:
            X_tmp = np.zeros((self.nb_sources, T))
            X_tmp[source1_id, :] = self.decay[0] * self.waveform
            src_ind = np.where(self.tri == source1_id)[0]
            sv = self.tri[src_ind]

            src2 = sv[np.where(sv != source1_id)]
            source2_id = []
            for x in src2:
                if x in source2_id:
                    pass
                else:
                    source2_id.append(x)

            source2_id = np.array(source2_id)
            source2_id.sort()
            X_tmp[source2_id, :] = self.decay[1] * self.waveform

            src2_ind = []
            for ss in source2_id:
                rows = np.where(self.tri == ss)[0]
                for row in rows:
                    if row in src2_ind:
                        pass
                    else:
                        src2_ind.append(row)

            src2_ind = np.array(src2_ind)
            src2_ind.sort()
            sv2 = self.tri[src2_ind]

            source3_id = []
            for source2 in source2_id:
                src3_tmp = sv2[np.where(sv2 != source2)]

                for element in src3_tmp:
                    if element in source3_id or element in source2_id or\
                            element == source1_id:
                        pass
                    else:
                        source3_id.append(element)

            source3_id = np.array(source3_id)
            source3_id.sort()
            X_tmp[source3_id, :] = self.decay[2] * self.waveform
            X = X + X_tmp

        return X