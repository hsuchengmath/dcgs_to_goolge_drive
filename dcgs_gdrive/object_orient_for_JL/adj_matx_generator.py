import numpy as np
import scipy.sparse as sp


'''
    def recommend(self, user, item, uid2index ,mat2index, rf_model, test_data):
        non_cold_start_user, non_cold_start_item, non_cold_start_index = list(), list(), list()
        cold_start_user, cold_start_item, cold_start_index = list(), list(), list()
        for i in range(len(user)):
            if user[i] in uid2index and item[i] in mat2index:
                non_cold_start_user.append(uid2index[user[i]])
                non_cold_start_item.append(mat2index[item[i]])
                non_cold_start_index.append(i)
            else:
                cold_start_user.append(user[i])
                cold_start_item.append(item[i])  
                cold_start_index.append(i)
        # non_cold_start part 
        non_cold_start_pred = list(self.model(non_cold_start_user, non_cold_start_item).cpu().detach().numpy())
        # cold_start part
        cold_start_pred = list()
        for i in range(len(cold_start_user)):
            dat = test_data[(test_data['client_sn']==cold_start_user[i]) & (test_data['MaterialID']==cold_start_item[i])]
            dat = np.array(dat)
            predictions = rf_model.predict(dat)
            predictions = predictions[0]   
            cold_start_pred.append(predictions)
        # intergrate
        predictions_list = list()
        for i in range(len(non_cold_start_pred)):
            predictions_list.append([non_cold_start_pred[i], non_cold_start_index[i]])
        for i in range(len(cold_start_pred)):
            predictions_list.append([cold_start_pred[i], cold_start_index[i]])
        predictions_list = sorted(predictions_list, key=lambda x:x[1])
        predictions_list = [element[0] for element in predictions_list]
        predictions = np.array(predictions_list)
        del self.model
        return predictions 
'''


class Adj_Matx_Generator:
    def __init__(self, user_num, item_num, user_item_inter):
        self.user_num = user_num
        self.item_num = item_num
        self.user_id_list = user_item_inter[0]
        self.item_id_list = user_item_inter[1]

    def build_u_v_matrix(self):
        self.R = sp.dok_matrix((self.user_num, self.item_num), dtype=np.float32)
        for i in range(len(self.user_id_list)):
            uid, iid = self.user_id_list[i], self.item_id_list[i]
            self.R[uid, iid] = 1
    
    def build_uv_uv_matrix(self):
        self.adj_mat = sp.dok_matrix((self.user_num + self.item_num, self.user_num + self.item_num), dtype=np.float32)
        self.adj_mat = self.adj_mat.tolil()
        R = self.R.tolil()
        self.adj_mat[:self.user_num, self.user_num:] = R
        self.adj_mat[self.user_num:, :self.user_num] = R.T
        self.adj_mat = self.adj_mat.todok()

    def mean_adj_single(self,adj):
        # D^-1 * A
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj = d_mat_inv.dot(adj)
        # norm_adj = adj.dot(d_mat_inv)
        return norm_adj.tocoo()

    def normalized_adj_single(self, adj):
        # D^-1/2 * A * D^-1/2 (normalized Laplacian)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        # bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
        bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        return bi_lap.tocoo()

    def check_adj_if_equal(self, adj):
        dense_A = np.array(adj.todense())
        degree = np.sum(dense_A, axis=1, keepdims=False)
        temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
        return temp

    def main(self):
        self.build_u_v_matrix()
        self.build_uv_uv_matrix()
        norm_adj_mat = self.mean_adj_single(self.adj_mat + sp.eye(self.adj_mat.shape[0]))
        mean_adj_mat = self.mean_adj_single(self.adj_mat)
        normal_adj_mat = self.normalized_adj_single(self.adj_mat + sp.eye(self.adj_mat.shape[0]))
        return self.adj_mat.tocsr(), norm_adj_mat.tocsr(), normal_adj_mat.tocsr()


if __name__ == '__main__':
    user_num, item_num = 5, 6
    user_id_list = [0,0,4,2,1,3,3,2,4]
    item_id_list = [1,2,5,4,3,0,1,1,4]
    user_item_inter = [user_id_list, item_id_list]
    amg_obj = Adj_Matx_Generator(user_num, item_num, user_item_inter)
    adj_matx, norm_adj_mat, mean_adj_mat  = amg_obj.main()
    print(norm_adj_mat)