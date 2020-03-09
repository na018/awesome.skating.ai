import numpy as np
from skatingAI.utils.utils import BodyParts


class HumanDistanceMap(object):
    def __init__(self):
        self.graph = {
            BodyParts.Head.name: [BodyParts.torso.name],
            BodyParts.RUpArm.name: [BodyParts.torso.name, BodyParts.RForeArm.name],
            BodyParts.RForeArm.name: [BodyParts.RUpArm.name, BodyParts.RHand.name],
            BodyParts.RHand.name: [BodyParts.RForeArm.name],
            BodyParts.LUpArm.name: [BodyParts.torso.name, BodyParts.LForeArm.name],
            BodyParts.LForeArm.name: [BodyParts.LUpArm.name, BodyParts.LHand.name],
            BodyParts.LHand.name: [BodyParts.LForeArm.name],
            BodyParts.torso.name: [BodyParts.Head.name, BodyParts.LUpArm.name, BodyParts.RUpArm.name,
                                   BodyParts.LThigh.name,
                                   BodyParts.RThigh.name],
            BodyParts.RThigh.name: [BodyParts.RLowLeg.name, BodyParts.torso.name],
            BodyParts.RLowLeg.name: [BodyParts.RFoot.name, BodyParts.RThigh.name],
            BodyParts.RFoot.name: [BodyParts.RThigh.name],
            BodyParts.LThigh.name: [BodyParts.torso.name, BodyParts.LLowLeg.name],
            BodyParts.LLowLeg.name: [BodyParts.LFoot.name, BodyParts.LThigh.name],
            BodyParts.LFoot.name: [BodyParts.LThigh.name]
        }
        self.weighted_distances = self._build_matrix()

    def _find_all_paths(self, start_vertex: str, end_vertex: str, path=[]):
        """ find all paths from start_vertex to
            end_vertex in graph """
        path = path + [start_vertex]
        if start_vertex == end_vertex:
            return [path]

        paths = []
        for vertex in self.graph[start_vertex]:
            if vertex not in path:
                extended_paths = self._find_all_paths(vertex,
                                                      end_vertex,
                                                      path)
                for p in extended_paths:
                    paths.append(p)
        return paths

    def _build_matrix(self):

        distance_map = []
        for a in self.graph:

            distances = []
            for i, b in enumerate(self.graph):
                parts = self._find_all_paths(a, b)[0]
                distances.append(len(parts) - 1)

            distance_map.append(distances)

        return self._matrix_formatations(distance_map)

    def _matrix_formatations(self, distance_map):
        distance_map = np.array(distance_map)
        distance_map = np.round((1- distance_map / distance_map.size).astype(np.float16),2)
        distance_map = np.insert(distance_map, (0), 1, axis=0)
        distance_map = np.insert(distance_map, (0), 1, axis=1)
        distance_map[0, 0] = 0

        return distance_map
