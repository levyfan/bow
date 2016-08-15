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

    public FeatureManager() throws IOException, URISyntaxException {
        this.features = new EnumMap<>(Feature.Type.class);
        this.features.put(Feature.Type.HSV, new Hsv());
        this.features.put(Feature.Type.CN, new Cn());
        this.features.put(Feature.Type.HOG, new Hog());
        this.features.put(Feature.Type.SILTP, new Siltp());
    }

    public void feature(BowImage bowImage) {
        for (Feature feature : features.values()) {
            feature.extract(bowImage);
        }

        // release memory
        bowImage.image = null;
        bowImage.image4 = null;
        bowImage.mask = null;
        bowImage.mask4 = null;
    }

    public void feature(BowImage bowImage, Feature.Type type) {
        features.get(type).extract(bowImage);

        // release memory
        bowImage.image = null;
        bowImage.image4 = null;
        bowImage.mask = null;
        bowImage.mask4 = null;
    }

    @Override
    public String toString() {
        return "FeatureManager{" +
                "features=" + features +
                '}';
    }
}
