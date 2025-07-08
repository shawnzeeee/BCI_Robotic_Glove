from mne.decoding import CSP
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

def get_csp_svm_pipeline():
    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
    svm = SVC(kernel='linear')
    return Pipeline([('csp', csp), ('svm', svm)])

def get_csp_lda_pipeline():
    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
    lda = LinearDiscriminantAnalysis()
    return Pipeline([('csp', csp), ('lda', lda)])

def get_csp_xgb_pipeline():
    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    return Pipeline([('csp', csp), ('xgb', xgb)])

def get_svm_pipeline():
    return SVC(kernel='linear')

def get_xgb_pipeline():
    return XGBClassifier(use_label_encoder=False, eval_metric='logloss')

def get_label_encoder():
    return LabelEncoder()
