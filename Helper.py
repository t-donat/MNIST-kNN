class Helper:

    @staticmethod
    def check_type(parameter, intended_type, name):
        if intended_type == callable and not callable(parameter):
            error = str(type(parameter))[8:-2]
            raise TypeError(f"Parameter {name}: expected callable, got {error} instead!")

        elif not isinstance(parameter, intended_type):
            error = str(type(parameter))[8:-2]

            if isinstance(intended_type, tuple):

                error = str(type(parameter))[8:-2]

                type_list = [str(item)[8:-2] for item in intended_type]
                display = " or ".join(type_list)

            else:
                display = str(intended_type)[8:-2]

            raise TypeError(f"Parameter {name}: Expected {display}, got {error} instead!")

    @staticmethod
    def check_ndim(array, dim, name):
        if array.ndim != dim:
            raise ValueError(f"{name}: Expected dim of {dim}, got {array.ndim} instead!")

    @staticmethod
    def below_zero(parameter, name):
        if parameter <= 0:
            raise ValueError(f"{name}: please enter a valid inout greater than 0!")
