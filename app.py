import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import shap, joblib, pickle
import matplotlib
from matplotlib import pyplot as plt

# def get_recommend_old(user_id, l_reg, full_item_df, train_feat, user_gen_feat, user_movie_watch):

#     X_test_user = (user_gen_feat.loc[[user_id]]
#                    .merge(full_item_df.loc[~full_item_df.index.isin(user_movie_watch[user_id])].reset_index(), how = 'cross')

#     )
#     movieId = X_test_user['movieId']
#     pred = l_reg.predict(X_test_user[train_feat])
#     X_test_user['pred'] = pred
#     rec_user = X_test_user.set_index(movieId).sort_values('pred',ascending = False).head()
#     return rec_user

def get_recommend(user_id, l_reg,l_rank, full_item_df, train_feat,rank_train_feat,  user_gen_feat, train_user_movie_watch):

    X_test_user = (user_gen_feat.loc[[user_id]]
                   .merge(full_item_df.loc[~full_item_df.index.isin(train_user_movie_watch[user_id])].reset_index(), how = 'cross')
    )
    movieId = X_test_user['movieId']
    # for c in catg_cols:
    #     X_test_user[c] = X_test_user[c].astype('category')
    pred = l_reg.predict_proba(X_test_user[train_feat])[:, 1]
    # X_test_user[X_train.columns] = l_reg.named_steps['s'].transform(X_test_user[X_train.columns])
    X_test_user['pred'] = pred
    X_test_rank = X_test_user.set_index(movieId).sort_values('pred',ascending = False).head(20)
    X_test_rank['timestamp_item_ts_std'] = X_test_rank['movi__ts_std']
    X_test_rank['rank_pred'] = l_rank.predict(X_test_rank[rank_train_feat]) * X_test_rank['pred']
    return X_test_rank.sort_values('rank_pred',ascending = False).head()

@st.cache_resource
def loading_pickles():
    user_model = joblib.load('./results/best_model/class/rank_class_part1/pipeline_user_clf_3.joblib')
    user_rank_model = joblib.load('./results/best_model/class/rank_class_part1/pipeline_rank_poly_V3.joblib')
    user_gen_feat= pd.read_pickle('./results/best_model/class/rank_class_part1/user_gen_feat.pkl')
    full_item_df= pd.read_pickle('./results/best_model/class/rank_class_part1/full_item_df.pkl')
    train_feat = pickle.load(open('./results/best_model/class/rank_class_part1/train_feat.pkl', 'rb'))
    rank_train_feat = pickle.load(open('./results/best_model/class/rank_class_part1/rank_train_feat.pkl', 'rb'))
    # user_movie_watch = pickle.load(open('./results/best_model/class/rank_class_part1/user_movie_watch.pkl', 'rb') )
    train_user_movie_watch = pickle.load(open('./results/best_model/class/rank_class_part1/train_user_movie_watch.pkl', 'rb') )
    rating_y_test = pd.read_pickle('./results/best_model/class/rank_class_part1/rating_y_test.pkl')
    rating_y_test['user_rated_testData'] = rating_y_test.pop('has_rated')
    return user_model,user_rank_model, user_gen_feat,full_item_df,train_feat, rank_train_feat, train_user_movie_watch,rating_y_test


user_model,user_rank_model, user_gen_feat,full_item_df,train_feat, rank_train_feat, train_user_movie_watch,rating_y_test = loading_pickles()

def main():

    st.title('Movie recommendation Web App')
    # with st.spinner("Loading model and libaries...."):
    
    uid = (st.text_input("Enter User Id"))
    if st.button("Get recommendations"):
        if uid.isdigit() and int(uid)  in train_user_movie_watch:
            uid = int(uid)
            st.text("User information")
            st.dataframe(user_gen_feat.loc[[uid]])
            with st.spinner("Generating prediction and explanations...."):
                # Get recommendation
                rec_user = get_recommend(uid, user_model[uid],user_rank_model[uid], full_item_df, train_feat, rank_train_feat, user_gen_feat, train_user_movie_watch)
                
                # Get preprocess and tree model 
                l_reg = user_model[uid]
                tree = l_reg.named_steps['reg']
                explainer = shap.TreeExplainer(tree)
                

                feat_cols_poly = l_reg.named_steps['p'].get_feature_names_out()
                X_test_poly = l_reg.named_steps['p'].transform(rec_user[train_feat])
                # Pass transformed data to tree
                X_test_poly = pd.DataFrame(X_test_poly, columns = feat_cols_poly, index = rec_user.index)
                rec_user = pd.concat([rec_user[rec_user.columns[~rec_user.columns.isin(train_feat)]],
                    X_test_poly
                ], axis=1)
                shap_values = explainer.shap_values(rec_user[feat_cols_poly])

                rec_user_display = rec_user.round(1).copy()

                for col in ['cast_1', 'cast_0', 'primary_lang']:
                    rec_user_display[f'enc_{col}'] = rec_user_display[col]
                
                movie_preds = rec_user_display.drop('movieId',1).merge(rating_y_test.loc[uid, ['user_rated_testData']], on= 'movieId', how = 'left')[['title','user_rated_testData', 'pred']].fillna(False)
                for r, (movieId, row) in enumerate(rec_user[['title','pred']].iterrows()):
                    movie = rec_user_display.iloc[r, :]
                    plt.clf()
                    shap.force_plot(explainer.expected_value[1], shap_values[1][r, :], movie[feat_cols_poly],feature_names = pd.Index(feat_cols_poly).str.replace(' ', ' -X- '), matplotlib=matplotlib, text_rotation=25, show = False)
                    plt.savefig(f"results/{r}.png",dpi=150, bbox_inches='tight')
                    st.dataframe(movie_preds.loc[[movieId]])
                    st.image(f"results/{r}.png")
            st.success("")
        else:
            st.success("No User id found")
    



if __name__=='__main__':
    main()