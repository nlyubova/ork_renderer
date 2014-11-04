/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2012, Willow Garage, Inc.
 *  Copyright (c) 2013, Vincent Rabaud
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include <object_recognition_renderer/renderer.h>
#include <object_recognition_renderer/utils.h>

#include <iostream>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

RendererIterator::RendererIterator(Renderer *renderer, size_t n_points)
    :
      n_points_(n_points),
      index_(0),
      renderer_(renderer),
      angle_min_(-80),
      angle_max_(80),
      angle_step_(40),
      angle_(angle_min_),
      radius_min_(0.4),
      radius_max_(0.8),
      radius_step_(0.2),
      radius_(radius_min_)
{
}

RendererIterator &
RendererIterator::operator++()
{
  angle_ += angle_step_;
  if (angle_ > angle_max_)
  {
    angle_ = angle_min_;
    radius_ += radius_step_;
    if (radius_ > radius_max_)
    {
      radius_ = radius_min_;
      ++index_;
    }
  }

  return *this;
}

void
RendererIterator::reinit()
{
  index_ = 0;
  angle_ = angle_min_;
  radius_ = radius_min_;
}

void
RendererIterator::render(cv::Mat &image_out, cv::Mat &depth_out, cv::Mat &mask_out, cv::Rect &rect_out)
{
  if (isDone())
    return;

  cv::Vec3d t, up;
  view_params(t, up);

  renderer_->lookAt(t(0), t(1), t(2), up(0), up(1), up(2));
  renderer_->render(image_out, depth_out, mask_out, rect_out);
}

void
RendererIterator::render_known(const cv::Vec3d t_in, cv::Mat &image_out, cv::Mat &depth_out, cv::Mat &mask_out, cv::Rect &rect_out)
{
  cv::Vec3d up, t(t_in);
  view_params_known(t, up);
  renderer_->lookAt(t(0), t(1), t(2), up(0), up(1), up(2));
  renderer_->render(image_out, depth_out, mask_out, rect_out);
}

cv::Vec3d
RendererIterator::T_known(const cv::Vec3d t_in)
{
  cv::Vec3d t(t_in), _up;
  view_params_known(t, _up);

  return -t;
}

/**
 * @return the rotation of the camera with respect to the current view point
 */
cv::Matx33d
RendererIterator::R() const
{
  cv::Vec3d up, t;
  view_params(t, up);
  normalize_vector(t(0),t(1),t(2));

  // compute the left vector
  cv::Vec3d y;
  y = up.cross(t);  // cross product
  normalize_vector(y(0),y(1),y(2));

  // re-compute the orthonormal up vector
  up = t.cross(y);  // cross product
  normalize_vector(up(0), up(1), up(2));

  cv::Mat R_full = (cv::Mat_<double>(3, 3) <<
                    t(0), t(1), t(2),
                    y(0), y(1), y(2),
                    up(0), up(1), up(2));
  cv::Matx33d R = R_full;
  R = R.t();

  return R;
}

/**
 * @return the rotation of the mesh with respect to the current view point
 */
cv::Matx33d
RendererIterator::R_obj() const
{
  cv::Vec3d up, t;
  view_params(t, up);
  normalize_vector(t(0),t(1),t(2));

  // compute the left vector
  cv::Vec3d y;
  y = up.cross(t);  // cross product
  normalize_vector(y(0),y(1),y(2));

  // re-compute the orthonormal up vector
  up = t.cross(y);  // cross product
  normalize_vector(up(0), up(1), up(2));

  cv::Mat R_full = (cv::Mat_<double>(3, 3) <<
                    -y(0), -y(1), -y(2),
                    -up(0), -up(1), -up(2),
                    t(0), t(1), t(2)
                    );

  cv::Matx33d R = R_full;
  R = R.t();

  return R.inv();
}

cv::Matx33d
RendererIterator::R_cam_known(const cv::Vec3d t_in) const
{
  cv::Vec3d up, t(t_in);
  view_params_known(t, up);

  // compute the left vector
  cv::Vec3d y;
  y = up.cross(t);  // cross product
  normalize_vector(y(0),y(1),y(2));

  // re-calculate the orthonormal up vector
  up = t.cross(y);  // cross product
  normalize_vector(up(0), up(1), up(2));

  //cv::Vec3d y = t.cross(up);
  cv::Mat R_full = (cv::Mat_<double>(3, 3) << t(0), t(1), t(2), y(0), y(1), y(2), up(0), up(1), up(2));
  cv::Matx33d R = R_full;

  return R;
}

cv::Matx33d
RendererIterator::R_obj_known(const cv::Vec3d t_in) const
{
  cv::Vec3d up, t(t_in);
  view_params_known(t, up); //, cv::Vec3d(1, 0, 0)

  cv::Vec3d t_normal(t(0), t(1), t(2));
  //normalize_vector(t_normal(0),t_normal(1),t_normal(2));
  normalize_vector(up(0),up(1),up(2));

  cv::Vec3d y = t_normal.cross(up);
  normalize_vector(y(0),y(1),y(2));
  normalize_vector(up(0),up(1),up(2));

  return cv::Matx33d(t(0), t(1), t(2),
                      y(0), y(1), y(2),
                      up(0), up(1), up(2));
}

float
RendererIterator::D_obj() const
{
  return radius_;
}

/**
 * @return the translation of the camera with respect to the current view point
 */
cv::Vec3d
RendererIterator::T() const
{
  cv::Vec3d t, _up;
  view_params(t, _up);

  return -t; //t
}

/**
 * @return the total number of templates that will be computed
 */
size_t
RendererIterator::n_templates() const
{
  return ((angle_max_ - angle_min_) / angle_step_ + 1) * n_points_ * ((radius_max_ - radius_min_) / radius_step_ + 1);
}

/**
 * @param T the translation vector
 * @param up the up vector of the view point
 */

void
RendererIterator::view_params(cv::Vec3d &T, cv::Vec3d &up, cv::Vec3d T_coincide) const
{
  float angle_rad = angle_ * CV_PI / 180.;

  // from http://www.xsi-blog.com/archives/115
  static float inc = CV_PI * (3 - sqrt(5));
  static float off = 2.0f / float(n_points_);

  float y = index_ * off - 1.0f + (off / 2.0f);
  float r = sqrt(1.0f - y * y);
  float phi = index_ * inc;
  float x = std::cos(phi) * r;
  float z = std::sin(phi) * r;

  //T = cv::Vec3d(x, y, z); //do not forget !!!!

  float lat = std::acos(z), lon; //z
  if ((fabs(std::sin(lat)) < 1e-5) || (fabs(y / std::sin(lat)) > 1)) //y
    lon = 0;
  else
    lon = std::asin(y / std::sin(lat)); //y

  x *= radius_; // * cos(lon) * sin(lat);
  y *= radius_; //float y = radius * sin(lon) * sin(lat);
  z *= radius_; //float z = radius * cos(lat);

  T = cv::Vec3d(x, y, z); //do not forget !!!!

  // Figure out the up vector
  float x_up = radius_ * std::cos(lon) * std::sin(lat - 1e-5) - x;
  float y_up = radius_ * std::sin(lon) * std::sin(lat - 1e-5) - y;
  float z_up = radius_ * std::cos(lat - 1e-5) - z;
  normalize_vector(x_up, y_up, z_up);

  // Figure out the third vector of the basis
  float x_right = -y_up * z + z_up * y;
  float y_right = x_up * z - z_up * x;
  float z_right = -x_up * y + y_up * x;
  normalize_vector(x_right, y_right, z_right);

  // Rotate the up vector in that basis
  float x_new_up = x_up * std::cos(angle_rad) + x_right * std::sin(angle_rad);
  float y_new_up = y_up * std::cos(angle_rad) + y_right * std::sin(angle_rad);
  float z_new_up = z_up * std::cos(angle_rad) + z_right * std::sin(angle_rad);
  up = cv::Vec3d(x_new_up, y_new_up, z_new_up);

  // compute the left vector
  cv::Vec3d l;
  l = up.cross(T);  // cross product
  normalize_vector(l(0),l(1),l(2));

  up = T.cross(l);  // cross product
  normalize_vector(up(0), up(1), up(2));
}

void
RendererIterator::view_params_known(cv::Vec3d &T, cv::Vec3d &up, cv::Vec3d T_coincide) const
{
  float angle_rad = angle_ * CV_PI / 180.;

  //------------------------------------
  // simple from original method

  //T *= -1;
  float x = -T(0);
  float y = -T(1);
  float z = -T(2);

  float lat = std::acos(z), lon; //z
  if ((fabs(std::sin(lat)) < 1e-5) || (fabs(y / std::sin(lat)) > 1)) //y
    lon = 0;
  else
    lon = std::asin(y / std::sin(lat)); //y

  x *= radius_; // * cos(lon) * sin(lat);
  y *= radius_; //float y = radius * sin(lon) * sin(lat);
  z *= radius_; //float z = radius * cos(lat);

  // Figure out the up vector
  float x_up = radius_ * std::cos(lon) * std::sin(lat - 1e-5) - x;
  float y_up = radius_ * std::sin(lon) * std::sin(lat - 1e-5) - y;
  float z_up = radius_ * std::cos(lat - 1e-5) - z;
  normalize_vector(x_up, y_up, z_up);

  // Figure out the third vector of the basis
  float x_right = -y_up * z + z_up * y;
  float y_right = x_up * z - z_up * x;
  float z_right = -x_up * y + y_up * x;
  normalize_vector(x_right, y_right, z_right);

  // Rotate the up vector in that basis
  float x_new_up = x_up * std::cos(angle_rad) + x_right * std::sin(angle_rad);
  float y_new_up = y_up * std::cos(angle_rad) + y_right * std::sin(angle_rad);
  float z_new_up = z_up * std::cos(angle_rad) + z_right * std::sin(angle_rad);
  up = cv::Vec3d(x_new_up, y_new_up, z_new_up);
  //cv::Vec3d f(x, y, z);

  // compute the left vector
  cv::Vec3d l;
  l = up.cross(T);  // cross product
  normalize_vector(l(0),l(1),l(2));

  up = T.cross(l);  // cross product
  normalize_vector(up(0), up(1), up(2));
}
