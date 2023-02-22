from sklearn import preprocessing
from data_utilities.Sample import Sample, Samples


def preprocess_samples(preprocessor, samples, is_test=False):
    samples_scaled = []
    iterator = iter(samples)

    sample = next(iterator)

    # we iterate over the samples and fit them
    while sample is not None:
        old_features = sample.get_features()
        name = sample.get_name()
        encoding = sample.get_encoding()
        # TODO: add scaling back
        # if is_test:
        #     scaled_sample = Sample(name=name, features=preprocessor.transform([old_features.reshape(-1)]), encoding=encoding)
        # else:
        #     scaled_sample = Sample(name=name, features=preprocessor.transform([old_features.reshape(-1)]), encoding=encoding)
        if is_test:
            scaled_sample = Sample(name=name, features=old_features,  encoding=encoding)
        else:
            scaled_sample = Sample(name=name, features= old_features , encoding=encoding)
        samples_scaled.append(scaled_sample)

        sample = next(iterator)

    return Samples(samples_scaled)

def preprocess_all_samples(train_samples, test_samples):
    train_samples = Samples(train_samples)
    test_samples = Samples(test_samples)

    preprocessor = preprocessing.StandardScaler()
    all_features = train_samples.get_features()
    preprocessor.fit(all_features[0] )

    train_samples_scaled = preprocess_samples(preprocessor, train_samples, is_test=False)
    test_samples_scaled = preprocess_samples(preprocessor, test_samples, is_test=True)

    return train_samples_scaled, test_samples_scaled
