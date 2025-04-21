import React from 'react';
import './Loading.css';

const Loading = () => {
  return (
    <div className="loading-container">
      {/* Replace '/logo.png' with the path to your logo */}
      <img src="/logo.png" alt="Loading Logo" className="loading-logo" />
      <div className="spinner"></div>
    </div>
  );
};

export default Loading;
