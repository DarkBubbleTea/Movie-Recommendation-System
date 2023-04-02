import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from collections import Counter
import pickle
import base64
from PIL import Image
from sklearn.preprocessing import LabelEncoder

datafile = 'movies.csv'

# trained_model = 'test_1_model.pkl'

image_folder = 'images'

def load_image(imagefile):
    img = Image.open(imagefile)
    return img

def main():
    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
    background-image: url("https://images.unsplash.com/photo-1501426026826-31c667bdf23d");
    background-size: cover;
    }}
    [data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)
    st.title('Hệ thống gợi ý phim :film_frames:')
    st.write('Chào mừng bạn đến với hệ thống gợi ý phim, tôi tạo ra cái này để giúp mọi người có thể tìm ra phim muốn xem dựa vào vài đặc điểm')
    st.write('Website này sử dụng một mô hình Machine learning đã được huấn luyện trên movies dataset từ movieslen')

    menu = ['Về tôi', 'Bộ dữ liệu', 'Gợi ý', 'EDA']
    choice = st.sidebar.selectbox('Menu', menu)
    st.write('---')

    if choice == 'Bộ dữ liệu':

        st.write('Bộ dữ liệu được show ở bên dưới')

        st.subheader('Bộ dữ liệu')
        df = pd.read_csv(datafile)
        st.dataframe(df)

        st.write('\n')

        st.write('Các cột có tương quan với nhau ở một mức độ nào đó theo phân tích. \n' + 'Các mối quan hệ tương quan này thường đợc thể hiện bằng cách sử dụng thước đo gọi là _correlation coefficient_, có thể thực hiện bằng phương thức .corr() trong Python.')

        imagefile = f'{image_folder}/asdasdasdasdasdasdasdasdasd.jpg'
        st.image(load_image(imagefile), width=285)

        st.write('Bộ dữ liệu chứa rất nhiều thông tin khác nhau như tên phim, thể loại, ngôn ngữ, độ phổ biến,...')
        st.write('Để thuận tiện cho việc xây dựng mô hình, tôi sẽ chỉ chọn ra các features như correllation bên trên và sau đó hiện ra các kết quả phù hợp với các lựa chọn')

    elif choice == 'Gợi ý':
        st.subheader('Gợi ý phim')
        ori_cat_list = ['en', 'Others']
        popu_cat_list = ['Very Low', 'Low', 'Medium', 'High']
        run_cat = ['Very Short', 'Short', 'Medium', 'Long']
        sta_cat = ['Released', 'Post production']
        original_language_cat = st.selectbox('Chọn ngôn ngữ gốc của phim, Others là các ngôn ngữ khác ngoài tiếng anh', ori_cat_list)
        popularity_cat = st.selectbox('Mức độ phổ biến, từ rất thấp đến cao', popu_cat_list)
        runtime_cat = st.selectbox('Độ dài của phim, dưới 30 phút là rất ngắn và trên 120 phút là dài', run_cat)
        status = st.selectbox('Tình trạng của phim, đã ra mắt hay chưa', sta_cat)
        vote_average = st.slider('Số điểm trung bình được đánh giá cho phim bạn muốn tìm', min_value=0.0, max_value=10.0, step=0.1)

        lookup = pd.read_csv('lookup.csv')
        le = LabelEncoder()
        lookup[['original_language_cat', 'popularity_cat', 'runtime_cat', 'status']] = lookup[
            ['original_language_cat', 'popularity_cat', 'runtime_cat', 'status']].apply(le.fit_transform)

        def movies_recommend(language, popular, runtime, status, vote):
            with open('test_1_model.pkl', 'rb') as f:
                model = pickle.load(f)
            arr = []
            for col, val in zip(['original_language_cat', 'popularity_cat', 'runtime_cat', 'status', 'vote_average'],
                                [language, popular, runtime, status, vote]):
                if val in le.classes_:
                    arr.append(le.transform([val])[0])
                else:
                    arr.append(-1)
            arr.append(vote)
            arr = np.array(arr).reshape(1, -1)
            pred = model.predict(arr)

            # Load the lookup dataframe
            lookup = pd.read_csv('lookup.csv')

            cluster = pred[0]
            mask = lookup['cluster'] == cluster
            return lookup[mask].sample(5)

        trigger = st.button('Gợi ý')
        if trigger:
            result = movies_recommend(original_language_cat, popularity_cat, runtime_cat, status, vote_average)
            st.write(result)

    elif choice == 'EDA':
        st.title('Movielens Exploratory Data Analysis')
        st.text('Đây là trang về khám phá dữ liệu Movieslen')

        st.header('Data Cleansing')
        df = pd.read_csv(datafile)
        st.write('Đầu tiên mình sẽ bỏ đi những cột không cần thiết')
        st.write('\n')
        drop_col = ['index', 'original_title', 'status', 'homepage' , 'tagline','keywords', 'cast', 'overview' , 'id']
        df.drop(drop_col ,  axis=1, inplace=True)
        st.write(f'Đây là các cột cần loại bỏ: {drop_col}')
        st.write('\n')

        st.subheader('Dữ liệu sau khi loại bỏ các cột không cần thiết')
        st.write(df)
        st.write('\n')

        st.subheader('Số dòng và cột')
        rows, cols = df.shape
        st.write(f'Số dòng: {rows}')
        st.write(f'Số cột: {cols}')
        df['release_date'] = pd.to_datetime(df['release_date'])
        df['director'] = df['director'].fillna('Unknown')
        df['genres'] = df['genres'].fillna('Unknown')
        df['release_year'] = df['release_date'].dt.year
        df['release_year'] = df['release_year'].fillna(0).astype(int)
        df = df.dropna()
        st.write('\n')
        st.write('\n')

        st.header('Visualize')
        # Most expensive
        top_10_movies = df.sort_values('budget', ascending=False).head(10)
        fig1 = px.bar(top_10_movies, x='title', y='budget', color='title', title='Top 10 Phim Kinh Phí Cao Nhất')
        st.write(fig1)

        # Most revenue
        fig2 = px.bar(df.sort_values('revenue', ascending=False).head(10), x='title', y='revenue', color='title',
                      title='Top 10 Phim Có Lợi Nhuận Cao Nhất')
        st.write(fig2)

        # Most long d*ck
        runtime_180_plus = df[df['runtime'] > 180][['title', 'runtime']].sort_values(by='runtime', ascending=True)
        fig3 = px.bar(runtime_180_plus, x='title', y='runtime',
                      title='Những Phim có thời lượng dài hơn 190 phút(3 tiếng)', width=1000, height=700)
        st.write(fig3)

        # Phim add
        col = "genres"
        categories = " ".join(df['genres']).split(" ")
        counter_list = Counter(categories).most_common(50)
        labels = [_[0] for _ in counter_list][::-1]
        values = [_[1] for _ in counter_list][::-1]
        trace1 = go.Bar(y=labels, x=values, orientation="h", name="TV Shows", marker=dict(color="#a678de"))
        data = [trace1]
        layout = go.Layout(title="Thể loại được thêm vào trong những năm qua", legend=dict(x=0.1, y=1.1, orientation="h"))
        fig4 = go.Figure(data, layout=layout)
        st.write(fig4)

        # By year release
        col = "release_year"
        vc2 = df[col].value_counts().reset_index()
        vc2 = vc2.rename(columns={col: "count", "index": col})

        vc2['percent'] = vc2['count'].apply(lambda x: 100 * x / sum(vc2['count']))
        vc2 = vc2.sort_values(col)
        trace2 = go.Bar(x=vc2[col], y=vc2["count"], name="Movies", marker=dict(color="#6ad49b"))
        data = [trace2]
        layout = go.Layout(title="Năm phát hành",
                           legend=dict(x=0.1, y=1.1, orientation="h"))
        fig5 = go.Figure(data, layout=layout)
        st.write(fig5)

        # Top 10 qualified score
        C = df['vote_average'].mean()
        m = df['vote_count'].quantile(0.9)
        q_movies = df.copy().loc[df['vote_count'] >= m]

        def weighted_rating(x, m=m, C=C):
            v = x['vote_count']
            R = x['vote_average']
            # Tính toán dựa trên công thức IMDB
            return (v / (v + m) * R) + (m / (m + v) * C)

        # Tạo feature score và dùng weighted_rating
        q_movies['score'] = q_movies.apply(weighted_rating, axis=1)
        q_movies = q_movies.sort_values('score', ascending=False)
        fig6 = px.pie(q_movies.head(10), values='score', names='title', title='Top 10 Phim xếp theo Qualified Score')
        st.write(fig6)

        pop = df.sort_values('popularity', ascending=False)
        plt.figure(figsize=(12, 4))
        plt.barh(pop['title'].head(6), pop['popularity'].head(6), align='center',
                 color='skyblue')
        plt.gca().invert_yaxis()
        plt.xlabel("Popularity")
        plt.title("Phim thịnh hành")
        st.pyplot(plt)

        # Top 15 by ratings
        qualified = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())][
            ['title', 'release_year', 'vote_count', 'vote_average', 'popularity', 'genres']]
        qualified['vote_count'] = qualified['vote_count'].astype('int')
        qualified['vote_average'] = qualified['vote_average'].astype('int')
        def weighted_rating(x):
            v = x['vote_count']
            R = x['vote_average']
            # Tính toán dựa trên công thức IMDB
            return (v / (v + m) * R) + (m / (m + v) * C)
        qualified['weighted_ratings'] = qualified.apply(weighted_rating, axis=1)
        qualified = qualified.sort_values('weighted_ratings', ascending=False).head(250)
        qualified.head(15)
        fig6 = px.bar(qualified.sort_values('weighted_ratings', ascending=False).head(15), x='title',
                      y='weighted_ratings', color='title', title='Top 15 Phim xếp theo Ratings')
        st.write(fig6)

        # Rec sys for viz
        s = df.apply(lambda x: pd.Series(x['genres']), axis=1).stack().reset_index(level=1, drop=True)
        s.name = 'genre'
        gen_md = df.drop('genres', axis=1).join(s)

        def build_chart(genre, percentile=0.85):
            df1 = gen_md[gen_md['genre'] == genre]
            vote_counts = df1[df1['vote_count'].notnull()]['vote_count'].astype('int')
            vote_averages = df1[df1['vote_average'].notnull()]['vote_average'].astype('int')
            C = vote_averages.mean()
            m = vote_counts.quantile(percentile)

            qualified = df1[(df1['vote_count'] >= m) & (df1['vote_count'].notnull()) & (df1['vote_average'].notnull())][
                ['title', 'release_year', 'vote_count', 'vote_average', 'popularity']]
            qualified['vote_count'] = qualified['vote_count'].astype('int')
            qualified['vote_average'] = qualified['vote_average'].astype('int')

            qualified['wr'] = qualified.apply(
                lambda x: (x['vote_count'] / (x['vote_count'] + m) * x['vote_average']) + (
                            m / (m + x['vote_count']) * C), axis=1)
            qualified = qualified.sort_values('wr', ascending=False).head(250)

            return qualified

        # Top 10 kinh dị
        fig7 = px.bar(build_chart('Horror').sort_values('wr', ascending=False).head(10), x='title', y='wr',
                      color='title', title='Top 10 Phim kinh dị xếp theo Ratings')
        st.write(fig7)

        # Top 10 dramu
        fig8 = px.bar(build_chart('Drama').sort_values('wr', ascending=False).head(10), x='title', y='wr',
                      color='title', title='Top 10 Phim chính kịch xếp theo Ratings')
        st.write(fig8)

        # Top 10 phim hài
        fig9 = px.bar(build_chart('Comedy').sort_values('wr', ascending=False).head(10), x='title', y='wr',
                      color='title', title='Top 10 Phim hài xếp theo Ratings')
        st.write(fig9)

        # Lợi nhuận từ phim tính theo năm
        grouped = df.groupby('release_year')['revenue'].sum().reset_index()
        fig10 = px.bar(grouped, x='release_year', y='revenue', color='release_year',
                       title='Lợi nhuận từ phim tính theo năm')
        st.write(fig10)

        # Kinh phí cho phim tính theo năm
        grouped = df.groupby('release_year')['budget'].sum().reset_index()
        fig11 = px.bar(grouped, x='release_year', y='budget', color='release_year', title='Kinh phí cho phim tính theo năm')
        st.write(fig11)

        # Doanh thu vs ngân sách
        fig12 = px.scatter(df, x='budget', y='revenue', color='release_year', title='Doanh thu vs Ngân sách(năm)')
        st.write(fig12)

        groupby_lang = df.groupby('original_language')['original_language'].count().reset_index(name='count').sort_values('count', ascending=False).iloc[:5]
        # Top 5 theo ngôn ngữ
        fig13 = px.bar(groupby_lang, x='original_language', y='count', color='original_language',
                       title='Movies by Language (Top 5)')
        st.write(fig13)

        # Top 10 theo thể loại
        df_top10 = df['genres'].value_counts().head(10).reset_index()
        fig14 = px.bar(df_top10, x='index', y='genres',
                       title='Top 10 thể loại phim',
                       labels={'index': 'Genre', 'genres': 'Number of Movies'},
                       color_discrete_sequence=['#648FFF'])
        st.write(fig14)

        # Phân tích theo năm phát hành
        plt.figure(figsize=(10, 10))
        sns.set(style="darkgrid")
        ax = sns.countplot(y="release_year", data=df, palette="coolwarm",
                           order=df['release_year'].value_counts().index[0:15])
        plt.title('Phân tích theo năm phát hành phim', fontsize=8, fontweight='bold', color='black')
        st.pyplot(plt)
        st.write('\n')

    elif choice == 'Về tôi':
        st.subheader('Về tôi')
        st.write('Phần mềm này được viết bởi tôi, một sinh viên năm 4 Hutech để giúp mọi người tìm kiếm phim theo nhu cầu dễ dàng hơn')
        st.write('\n')
        st.write('Tôi đang trong quá trình hoàn thiện phần mềm, trong quá trình sử dụng nếu có thắc mắc gì xin đừng ngại đặt câu hỏi')
        st.write('\n')
        st.write('Cảm ơn!')

if __name__ == "__main__":
  main()