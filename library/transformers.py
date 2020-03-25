
class ImportantAttributeSelector(BaseEstimator, TransformerMixin):
    # This Transformer helpes in selecting top attributes having linear correlation and eliminating a few

    def __init__(self, target_attribute):
        self.target_attribute = target_attribute
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, attr_count):

        # calculate correlation matrix
        corr_matrix = X.corr()

        # project the target attribute
        target_corr_matrix = corr_matrix[self.target_attribute]

        # target_corr_matrix represents linear relationship in the range of [-1, 1]
        # coefficients close to 0, are having less to no linear correlation

        # convert all the values to absolute and sort in ascending order
        tr_target_corr_matrix = list(map(lambda item: (abs(item[1]), item[0]), target_corr_matrix.items()))
        tr_target_corr_matrix.sort(reverse=True)

        # create the list of attributes and return the required attribute
        attr_set = list(map(lambda item: item[1], tr_target_corr_matrix))
        return attr_set[:attr_count]

