# Class that indicates where in the script it believes
# we are
class Locator:
    def __init__(self):
        # Initialize the current index to 0
        self.current_index = 0

        # Initialize an empty list to store the last n indices
        self.previous_indices = []

        # Initialize a list with the first index as the last best index
        self.last_best_indices = [0]

        # Set the number of previous indices to consider for location confidence to 3
        self.number_of_previous_indices = 3

    def next_index(self, index):
        # Append the current index to the list of previous indices
        self.previous_indices.append(index)

        # Check if the number of previous indices is less than 3
        # as 3 are needed to confirm location confidence
        if len(self.previous_indices) < 3:
            # Increment the current index by 1
            self.current_index += 1
        else:
            # Check if the difference between the second last and third last indices
            # is between 0 and 10, and the difference between the last and second last
            # indices is also between 0 and 10
            if (
                0 < self.previous_indices[-2] - self.previous_indices[-3] < 10
                and 0 < self.previous_indices[-1] - self.previous_indices[-2] < 10
            ):
                # Set the current index to the provided index
                self.current_index = index

                # Append the provided index to the list of last best indices
                self.last_best_indices.append(index)
            else:
                # Increment the current index by 1
                self.current_index += 1

        # Print the current index
        print("Current Index: {}".format(self.current_index))

        # Check if the provided index is within a range based on the current index
        return self.current_index - 5 < index < self.current_index + 40
