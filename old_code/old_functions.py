    def rewire_old(self, alpha, beta):

        neighbors = list(self.model.graph.neighbors(self.unique_id))
        if neighbors:
            d_edge = self.random.choice(neighbors)
            
            subgraph1 = nx.ego_graph(self.model.graph, 0 ,radius=2)
            subgraph2 = nx.ego_graph(self.model.graph, 0 ,radius=1)
            subgraph1.remove_nodes_from(subgraph2.nodes())
            second_order_neighbors= list(subgraph1.nodes())

            # probability of rewiring is proportional to total payoff difference (homophily)
            if second_order_neighbors:
                payoff_diff = []
                for second_neighbor in second_order_neighbors:
                    second_neigh_pay = self.model.schedule.agents[second_neighbor]
                    payoff_diff.append(np.abs(self.totPayoff - second_neigh_pay.totPayoff))
                pay_diff = np.array(payoff_diff)
                
                # limit pay_diff to 600 such that exp(600) does not overflow
                limit = 600
                pay_diff[(alpha*(pay_diff-beta)) > limit] = limit

                P_con = 1/(1+np.exp(alpha*(pay_diff-beta)))
                P_con = np.nan_to_num(P_con)
                if sum(P_con) == 0:
                    P_con = P_con + 1/len(P_con)                 
                # Calculate the sum of P_con
                total_prob = sum(P_con)

                # Check if the sum of P_con is close enough to 1 within a certain tolerance
                if not np.isclose(total_prob, 1, rtol=1e-9, atol=1e-9):
                    # Adjust P_con by dividing it by the sum of its elements and adding the remaining difference to one of the probabilities
                    P_con = P_con / total_prob
                    P_con[-1] += 1 - sum(P_con)

                # Normalize P_con by dividing it by the sum of its elements
                P_con = P_con / sum(P_con)



                add_neighbor = np.random.choice(second_order_neighbors, p=P_con)
                self.model.graph.add_edge(self.unique_id, add_neighbor)



            del_neigh_pay = self.model.schedule.agents[d_edge]
            if self.model.graph.has_edge(self.unique_id, d_edge) and self.totPayoff > del_neigh_pay.totPayoff:
                    if self.model.graph.degree(self.unique_id) > 1:
                        if self.model.graph.degree(d_edge) > 1:
                            self.model.graph.remove_edge(self.unique_id, d_edge)