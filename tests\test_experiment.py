from care_priority.experiment import dataset,evaluate
def test_shape():
 x,g,y=dataset(20,2);assert x.shape==(20,8) and len(g)==len(y)==20
def test_metrics(): assert all(0<=v<=1 for v in evaluate(2).values())
