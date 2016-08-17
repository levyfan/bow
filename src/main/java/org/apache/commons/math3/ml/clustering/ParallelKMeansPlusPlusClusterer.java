/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.commons.math3.ml.clustering;

import org.apache.commons.math3.exception.ConvergenceException;
import org.apache.commons.math3.exception.MathIllegalArgumentException;
import org.apache.commons.math3.exception.NumberIsTooSmallException;
import org.apache.commons.math3.ml.distance.DistanceMeasure;
import org.apache.commons.math3.ml.distance.EuclideanDistance;
import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.util.MathUtils;

import java.util.Collection;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Collectors;

/**
 * Clustering algorithm based on David Arthur and Sergei Vassilvitski k-means++ algorithm.
 *
 * @param <T> type of the points to cluster
 * @see <a href="http://en.wikipedia.org/wiki/K-means%2B%2B">K-means++ (wikipedia)</a>
 * @since 3.2
 */
public class ParallelKMeansPlusPlusClusterer<T extends Clusterable> extends KMeansPlusPlusClusterer<T> {

    /**
     * Build a clusterer.
     * <p>
     * The default strategy for handling empty clusters that may appear during
     * algorithm iterations is to split the cluster with largest distance variance.
     * <p>
     * The euclidean distance will be used as default distance measure.
     *
     * @param k the number of clusters to split the data into
     */
    public ParallelKMeansPlusPlusClusterer(final int k) {
        this(k, -1);
    }

    /**
     * Build a clusterer.
     * <p>
     * The default strategy for handling empty clusters that may appear during
     * algorithm iterations is to split the cluster with largest distance variance.
     * <p>
     * The euclidean distance will be used as default distance measure.
     *
     * @param k             the number of clusters to split the data into
     * @param maxIterations the maximum number of iterations to run the algorithm for.
     *                      If negative, no maximum will be used.
     */
    public ParallelKMeansPlusPlusClusterer(final int k, final int maxIterations) {
        this(k, maxIterations, new EuclideanDistance());
    }

    /**
     * Build a clusterer.
     * <p>
     * The default strategy for handling empty clusters that may appear during
     * algorithm iterations is to split the cluster with largest distance variance.
     *
     * @param k             the number of clusters to split the data into
     * @param maxIterations the maximum number of iterations to run the algorithm for.
     *                      If negative, no maximum will be used.
     * @param measure       the distance measure to use
     */
    public ParallelKMeansPlusPlusClusterer(final int k, final int maxIterations, final DistanceMeasure measure) {
        this(k, maxIterations, measure, new JDKRandomGenerator());
    }

    /**
     * Build a clusterer.
     * <p>
     * The default strategy for handling empty clusters that may appear during
     * algorithm iterations is to split the cluster with largest distance variance.
     *
     * @param k             the number of clusters to split the data into
     * @param maxIterations the maximum number of iterations to run the algorithm for.
     *                      If negative, no maximum will be used.
     * @param measure       the distance measure to use
     * @param random        random generator to use for choosing initial centers
     */
    public ParallelKMeansPlusPlusClusterer(final int k, final int maxIterations,
                                           final DistanceMeasure measure,
                                           final RandomGenerator random) {
        this(k, maxIterations, measure, random, EmptyClusterStrategy.LARGEST_VARIANCE);
    }

    /**
     * Build a clusterer.
     *
     * @param k             the number of clusters to split the data into
     * @param maxIterations the maximum number of iterations to run the algorithm for.
     *                      If negative, no maximum will be used.
     * @param measure       the distance measure to use
     * @param random        random generator to use for choosing initial centers
     * @param emptyStrategy strategy to use for handling empty clusters that
     *                      may appear during algorithm iterations
     */
    public ParallelKMeansPlusPlusClusterer(final int k, final int maxIterations,
                                           final DistanceMeasure measure,
                                           final RandomGenerator random,
                                           final EmptyClusterStrategy emptyStrategy) {
        super(k, maxIterations, measure, random, emptyStrategy);
    }

    /**
     * Runs the K-means++ clustering algorithm.
     *
     * @param points the points to cluster
     * @return a list of clusters containing the points
     * @throws MathIllegalArgumentException if the data points are null or the number
     *                                      of clusters is larger than the number of data points
     * @throws ConvergenceException         if an empty cluster is encountered and the
     *                                      {@link #emptyStrategy} is set to {@code ERROR}
     */
    @Override
    public List<CentroidCluster<T>> cluster(final Collection<T> points)
            throws MathIllegalArgumentException, ConvergenceException {

        // sanity checks
        MathUtils.checkNotNull(points);

        // number of clusters has to be smaller or equal the number of data points
        if (points.size() < k) {
            throw new NumberIsTooSmallException(points.size(), k, false);
        }

        // create the initial clusters
        List<CentroidCluster<T>> clusters = chooseInitialCenters(points);

        // create an array containing the latest assignment of a point to a cluster
        // no need to initialize the array, as it will be filled with the first assignment
        int[] assignments = new int[points.size()];
        assignPointsToClusters(clusters, points, assignments);

        // iterate through updating the centers until we're done
        final int max = (maxIterations < 0) ? Integer.MAX_VALUE : maxIterations;
        for (int count = 0; count < max; count++) {
            AtomicBoolean emptyCluster = new AtomicBoolean(false);
            final List<CentroidCluster<T>> finalClusters = clusters;

            List<CentroidCluster<T>> newClusters = finalClusters.parallelStream().map(cluster -> {
                final Clusterable newCenter;
                if (cluster.getPoints().isEmpty()) {
                    newCenter = getClusterable(finalClusters);
                    emptyCluster.set(true);
                } else {
                    newCenter = centroidOf(cluster.getPoints(), cluster.getCenter().getPoint().length);
                }
                return new CentroidCluster<T>(newCenter);
            }).collect(Collectors.toList());

            int changes = assignPointsToClusters(newClusters, points, assignments);
            clusters = newClusters;

            // if there were no more changes in the point-to-cluster assignment
            // and there are no empty clusters left, return the current clusters
            if (changes == 0 && !emptyCluster.get()) {
                return clusters;
            }
        }
        return clusters;
    }
}
