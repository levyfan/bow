package com.github.levyfan.reid.feature;

import com.github.levyfan.reid.BowImage;

import java.io.IOException;
import java.net.URISyntaxException;
import java.util.EnumMap;
import java.util.Map;

/**
 * @author fanliwen
 */
public class FeatureManager {

    private Map<Feature.Type, Feature> features;

    private Feature.Type[] featureTypes;

    public FeatureManager() throws IOException, URISyntaxException {
        this.features = new EnumMap<>(Feature.Type.class);
        this.features.put(Feature.Type.HSV, new Hsv());
        this.features.put(Feature.Type.CN, new Cn());
        this.features.put(Feature.Type.HOG, new Hog());
        this.features.put(Feature.Type.SILTP, new Siltp());
//        this.features.put(Feature.Type.OPQ_1, new Opq.Opq_1());
//        this.features.put(Feature.Type.OPQ_2, new Opq.Opq_2());
//        this.features.put(Feature.Type.OPQ_3, new Opq.Opq_3());
//        this.features.put(Feature.Type.OPQ_4, new Opq.Opq_4());

        featureTypes = new Feature.Type[]{
                Feature.Type.HSV, Feature.Type.CN, Feature.Type.HOG, Feature.Type.SILTP,
//                Feature.Type.OPQ_1, Feature.Type.OPQ_2, Feature.Type.OPQ_3, Feature.Type.OPQ_4
        };
    }

    public void feature(BowImage bowImage) {
        for (Feature.Type type : featureTypes) {
            features.get(type).extract(bowImage);
        }

        // release memory
//        bowImage.image = null;
//        bowImage.image4 = null;
//        bowImage.mask = null;
//        bowImage.mask4 = null;
    }

    public void feature(BowImage bowImage, Feature.Type type) {
        features.get(type).extract(bowImage);

        // release memory
//        bowImage.image = null;
//        bowImage.image4 = null;
//        bowImage.mask = null;
//        bowImage.mask4 = null;
    }

    @Override
    public String toString() {
        return "FeatureManager{" +
                "features=" + features +
                '}';
    }
}
