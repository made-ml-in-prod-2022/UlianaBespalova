from random import randrange, choice
import numpy as np


class HeartCleverlandGenerator:

    def generate_data(self):
        return {
        'age': str(randrange(30, 90)),
        'sex': str(choice([0, 1])),
        'cp': str(choice([0, 1, 2, 3])),
        'restbps': str(randrange(100, 150)),
        'chol': str(randrange(200, 350)),
        'fbs': str(choice([0, 1])),
        'restecg': str(choice([0, 1, 2])),
        'thalach': str(randrange(150, 350)),
        'exang': str(choice([0, 1])),
        'oldpeak': str(randrange(1, 4)),
        'slope': str(choice([0, 1, 2])),
        'ca': str(choice([0, 1, 2, 3])),
        'thal': str(choice([0, 1, 2])),
        'condition': str(choice([0, 1]))
        }

    def create_row(self, sep=','):
        data = self.generate_data()
        title_row = sep.join(data.keys())
        row = sep.join(list(data.values()))
        return [title_row, row]

    def generate_heart_cleveland_csv(self, num_elements):
        dataset = []
        for i in range(num_elements):
            title_row, row = self.create_row()
            if len(dataset) == 0:
                dataset.append(title_row)
            dataset.append(row)
        return np.array(dataset)


def main():
    generator = HeartCleverlandGenerator()
    dataset = generator.generate_heart_cleveland_csv(1000)
    print(dataset)
    np.savetxt("test_data.csv", dataset, delimiter="\n", fmt='%s')


if __name__ == "__main__":
    main()
