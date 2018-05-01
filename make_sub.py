import pandas as pd
from datetime import datetime

path = 'F:/Nerdy Stuff/Kaggle/Talking data/data/'
time_now = datetime.now().strftime("%d_%m_%y_%H_%m")

output_path = 'F:/Nerdy Stuff/Kaggle submissions/TalkingData/'

test = pd.read_csv(path + 'test.csv')
preds = pd.read_csv('C:/Users/Evan/PycharmProjects/StackNet/query_pred.csv', header=None)

pred_set = pd.DataFrame(data={'click_id': test['click_id'],
                              'project_is_approved': preds.loc[:, 1]})

pred_set.to_csv(output_path + time_now + '.csv', index=False)
